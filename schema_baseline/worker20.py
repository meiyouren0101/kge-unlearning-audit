"""One phase or frozen evaluation per child process. Scientific kernels stay unchanged."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse, hashlib, json, math, sys, time, shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from igpu import bootstrap
import common as c, graph_data as gd, legacy_runtime as rt, rank_kernels as rk, probe_kernels as pk
import gpu_wait as gpu, matrix20 as m, retry_bridge as rb
from igpu import audit, training as tr
from baselines import zf_core as zf
import torch
import numpy as np

def config(arm):
    v=m.read(m.ROOT/'configs'/(m.CONFIG.get(arm,arm)+'.json'))
    return dict(v,name=arm,diagnostic_probe=False)
def configure():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def sealed(session,name):
    f=session/'stages'/name
    v=m.read(f/'commit.json')
    m.check(v['status']=='COMMITTED','Unsealed input stage')
    for p,h in v['files'].items():m.verify_file(f/p,h)
    return f,m.read(f/'result.json'),v
def payload(path,h,history=False):
    x=rt.load_checkpoint_checked(path,h,True)
    if history:
        x=dict(x,metadata=dict(x['metadata'],kind='P'))
    return x
def start(session,tag,seed,spec_key):
    if spec_key is None:
        p,h=m.pin_p(tag,seed);return payload(p,h,True),h
    f,r,seal=sealed(session,spec_key)
    m.check(r['status']=='COMPLETE' and r['checkpoint_sha256']==seal['files']['final.pt'],'No valid input endpoint')
    h=seal['files']['final.pt'];return payload(f/'final.pt',h),h
def ranks(model,qs,index,device,batch,stream=None,one_filter=False):
    rows=[]
    with torch.no_grad():
        for i in range(0,len(qs),batch):
            q=qs[i:i+batch]
            out=rk.score(torch,model,q,index,device,('complete',) if one_filter else ('complete','legacy'))
            a=np.column_stack((q,out.get('legacy',out['complete']),out['complete'])).astype('<u4')
            if stream is not None:stream.write(a.tobytes())
            rows.append(a)
    return np.concatenate(rows) if rows else np.empty((0,5),dtype='<u4')
def make_zf(data,cfg,phase,seed,device,source):
    rt.seed_rng(seed)
    up,_=tr.model_api(cfg);args=tr.args_for(cfg,data,phase,seed,device)
    model=up.TransE(args,gd.kg(data,phase));rt.load_model_state(model,source['model'])
    model.eval();return up,model
def grid(session,tag,data,cfg,device):
    p,h=start(session,tag,0,None)
    f,_,_=sealed(session,'evaluate_P_0_p0')
    cohort=gd.at(data,0);ret=list(gd.queries(cohort['retained']));deleted=list(gd.queries(cohort['deleted_order']))
    order=sorted(np.random.Generator(np.random.PCG64(2514013)).choice(len(ret),10000,replace=False))
    sample=[ret[i] for i in order];old=np.fromfile(f/'ranks.u32le',dtype='<u4').reshape(-1,5)
    picked=old[len(deleted)+np.asarray(order)]
    m.check(np.array_equal(picked[:,:3],sample),'ZF grid P sample not paired')
    p_mrr=float((1/picked[:,4]).mean())
    up,model=make_zf(data,cfg,0,0,device,p)
    rb.record('source',0,model)
    ids,step,info=zf.correction(up,model,data,0,rt.schedule_seed(0,0),False)
    rb.ACTIVE.anchor('ihvp',rt.tensor_tree_hash(step))
    base=tuple(x.index_select(0,i).detach() for x,i in zip(model.embedding(),ids))
    index=gd.filters(data['triples']);rows=[]
    for eta in m.SCALES:
        row=dict(scale=eta,status='DIVERGED',retained_MRR=None,deleted_MRR=None,update_l2=None)
        zf.apply(model,ids,step,eta,base);norm=info['ihvp_l2']/eta
        if math.isfinite(norm):row['update_l2']=norm
        if all(bool(torch.isfinite(x).all()) for x in model.parameters()):
            try:
                rr=ranks(model,sample,index,device,16,one_filter=True)[:,4]
                dr=ranks(model,deleted,index,device,16,one_filter=True)[:,4]
            except ValueError as exc:
                if 'Nonfinite frozen score' not in str(exc):raise
            else:row.update(status='COMPLETE',retained_MRR=float((1/rr).mean()),deleted_MRR=float((1/dr).mean()))
        rows.append(row);rb.ACTIVE.anchor('eta/'+str(eta),rt.tensor_tree_hash(row));print('SCALE_RESULT',json.dumps(row),flush=True)
    scale,fallback=m.select_scale(rows,p_mrr)
    return dict(policy='grid',scale=scale,fallback=fallback,P_retained_MRR=p_mrr,grid=rows,sample_sha256=rt.tensor_tree_hash(sample),P_sha256=h,**info)

def evaluate(data,cfg,seed,phase,device,source,out,arm,extra_preferred=False):
    model=tr.load_model(data,cfg,seed,phase,device,source);rb.record('frozen',0,model)
    index=gd.filters(data['triples']);d=gd.at(data,phase)
    deleted=list(gd.queries(d['deleted_order']));qs=deleted+list(gd.queries(d['retained']))
    with (out/'ranks.u32le').open('xb') as f:
        a=ranks(model,qs,index,device,cfg['query_batch_size'],stream=f)
    info=dict(seed=seed,phase=phase,method=arm,queries=len(qs),n_deleted=len(deleted),
              query_sha256=hashlib.sha256(a[:,:3].tobytes()).hexdigest(),metrics=rk.summaries(a[:,3:],len(deleted)),
              cohort='full current retained population',preferred=[])
    if extra_preferred or arm in ('P','R','GraphDPO','GraphDPO-code','IGPU','ZF'):
        up,_=tr.model_api(cfg);rt.seed_rng(rt.schedule_seed(seed,phase))
        a0=tr.args_for(cfg,data,phase,seed,device);a0.unlearning_method='GraphDPO'
        ds=up.GraphDPOTrainBatchProcessor(a0,gd.kg(data,phase)).dataset
        facts=list(ds.pos_facts[::2]);heads=data['is_heads'][phase]
        pq=[(t,r+1,h) if side else (h,r,t) for (h,r,t),side in zip(facts,heads)]
        check_truth=set(data['triples'])
        with (out/'preferred.u32le').open('xb') as f:
            pr=ranks(model,pq,index,device,16,stream=f,one_filter=True)
        for pop,mask in (('all_original_preferred',np.ones(len(pq),dtype=bool)),('unobserved_original_preferred',np.asarray([x not in check_truth for x in facts],dtype=bool))):
            info['preferred'].extend(dict(population=pop,**r) for r in pk.metrics(pr[mask,4],pr[mask,:3]))
    c.write(out/'evaluation.json',info)
    return dict(queries=len(qs),query_sha256=info['query_sha256'])

def execute(data,cfg,device,session,out,spec):
    arm,seed,phase,mode,tag=[spec[k] for k in ('arm','seed','phase','mode','tag')]
    if mode=='grid':return grid(session,tag,data,cfg,device)
    if mode=='reference':
        end=tr.pretrain(data,cfg,seed,device,phase)
        expected=100*((2*len(gd.at(data,phase)['retained'])+1023)//1024)
        m.check(end['metadata']['updates']==expected,'R update budget differs')
        info={}
    else:
        source,h=start(session,tag,seed,spec.get('source',spec.get('previous')))
        rb.ACTIVE.anchor('input_checkpoint',h)
        info=dict(source_checkpoint_sha256=h)
        if mode=='evaluate':
            return {**info,**evaluate(data,cfg,seed,phase,device,source,out,arm)}
        if mode=='repair':
            budget=tr.stage_repair_steps(cfg,len(gd.at(data,phase)['retained']))
            end=tr.repair(data,cfg,seed,phase,device,source,budget)
        elif arm=='ZF':
            _,sel,_=sealed(session,'scale_ZF');eta=sel['scale']
            up,model=make_zf(data,cfg,phase,seed,device,source);rb.record('source',0,model)
            ids,delta,more=zf.correction(up,model,data,phase,rt.schedule_seed(seed,phase),False)
            rb.ACTIVE.anchor('ihvp',rt.tensor_tree_hash(delta));zf.apply(model,ids,delta,eta)
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):raise FloatingPointError('ZF nonfinite parameters')
            rb.record('corrected',1,model)
            end=dict(schema_version=1,model=rt.cpu_tree(model.state_dict()),optimizer=None,rng=rt.capture_rng(),
                     metadata=dict(kind='P',phase=phase,seed=seed,updates=1),loader_generator=None)
            info.update(eta=eta,**more)
        else:
            prev=None
            if spec.get('previous_delete'):
                prev,ph=start(session,tag,seed,spec['previous_delete']);rb.ACTIVE.anchor('previous_delete_Adam_input',ph)
            end,log=tr.delete(data,cfg,seed,phase,device,source,prev)
            for k in ('original_preferred','actual_positives'):log.pop(k,None)
            c.write(out/'training.json',log)
            if not all(bool(torch.isfinite(v).all()) for v in end['model'].values() if torch.is_tensor(v)):
                raise FloatingPointError('Nonfinite official state')
    h=rt.save_checkpoint_new(out/'final.pt',end)
    return dict(info,checkpoint_sha256=h,optimizer_steps=end['metadata']['updates'],kind=end['metadata']['kind'])

def main():
    p=argparse.ArgumentParser();p.add_argument('session',type=Path);p.add_argument('stage');a=p.parse_args()
    session=a.session;plan=m.read(session/'plan.json')
    spec=next(s for s in plan['stages'] if s['name']==a.stage)
    out=session/'stages'/a.stage;cfg=config(spec['arm']);t=time.monotonic()
    try:
        torch.cuda.set_device(0);configure();torch.cuda.reset_peak_memory_stats()
        data=m.graph(spec['tag'],plan['archive']);device=torch.device('cuda:0')
        with rb.Bridge(session,spec,out) as bridge:
            bridge.anchor('archive_sha256',m.digest(plan['archive']))
            try: result=dict(status='COMPLETE',**execute(data,cfg,device,session,out,spec))
            except FloatingPointError as exc:
                result=dict(status='DIVERGED',reason=str(exc))
            except ValueError as exc:
                if 'Nonfinite frozen score' not in str(exc):raise
                result=dict(status='DIVERGED',reason=str(exc))
            bridge.finish()
        torch.cuda.synchronize()
        c.write(out/'result.json',dict(spec,**result,seconds=time.monotonic()-t,gpu_peak_reserved_mib=gpu.torch_peak_mib()))
    except BaseException as exc:
        if not gpu.is_cuda_oom(exc):raise
        try:snapshot=gpu.query()
        except Exception:snapshot=None
        gpu.oom_marker(out,spec,exc,snapshot=snapshot)
    finally:gpu.clear_stage_cache(out)
if __name__=='__main__':main()
