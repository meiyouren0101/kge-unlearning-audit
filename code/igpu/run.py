"""Serial configured four-request run. Writes only a new user-chosen output."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
_device_index=sys.argv.index('--device')+1 if '--device' in sys.argv else len(sys.argv)
if _device_index>=len(sys.argv) or sys.argv[_device_index]=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
from . import bootstrap
from pathlib import Path
import argparse,copy,json,traceback
import numpy as np
import torch
from . import training as tr,data as dat
import common as c,graph_data as gd,legacy_runtime as rt,rank_kernels as rk,probe_kernels as pk

def raw_ranks(model,queries,data,cfg,batch_size=None):
    result=[];index=gd.filters(data['triples'])
    batch=batch_size or cfg['query_batch_size']
    with torch.no_grad():
        for i in range(0,len(queries),batch):
            qs=queries[i:i+batch];r=rk.score(torch,model,qs,index,next(model.parameters()).device)
            result.extend((*q,a,b) for q,a,b in zip(qs,r['legacy'],r['complete']))
    return np.asarray(result,dtype='<u4').reshape(-1,5)

def original_preferred(data,cfg,seed,phase,device):
    up,_=tr.model_api(cfg);rt.seed_rng(rt.schedule_seed(seed,phase))
    a=tr.args_for(cfg,data,phase,seed,device);a.unlearning_method='GraphDPO'
    proc=up.GraphDPOTrainBatchProcessor(a,gd.kg(data,phase))
    return dict(facts=proc.dataset.pos_facts[::2],is_heads=data['is_heads'][phase])

def evaluate(payload,data,cfg,seed,phase,device,out,preferred,queries=None,preferred_indices=None):
    model=tr.load_model(data,cfg,seed,phase,device,payload);state=rt.tensor_tree_hash(model.state_dict());d=gd.at(data,phase)
    if queries is None:
        deleted=list(gd.queries(d['deleted_order']));queries=deleted+list(gd.queries(d['retained']));nd=len(deleted)
    else:queries,nd=queries
    ranks=np.concatenate([raw_ranks(model,queries[:nd],data,cfg),raw_ranks(model,queries[nd:],data,cfg)])
    ranks.tofile(out/'ranks.u32le')
    if cfg['evaluation']=='sampled':
        import sampled_metrics
        stats=sampled_metrics.summaries(ranks[:,3:].tolist(),nd,queries)
    else:stats=rk.summaries(ranks[:,3:].tolist(),nd)
    chosen=range(len(preferred['facts'])) if preferred_indices is None else preferred_indices
    facts=[tuple(preferred['facts'][i]) for i in chosen];heads=[preferred['is_heads'][i] for i in chosen]
    pq=[(t,r+1,h) if head else (h,r,t) for (h,r,t),head in zip(facts,heads)]
    pr=raw_ranks(model,pq,data,cfg,batch_size=16 if cfg['evaluation']=='full' else 4);pr.tofile(out/'preferred.u32le');truth=set(data['triples'])
    pref=[]
    for population,mask in (('all_original_preferred',np.ones(len(pq),dtype=bool)),('unobserved_original_preferred',np.array([q not in truth for q in facts]))):
        for row in pk.metrics(pr[mask,4],pr[mask,:3]):pref.append(dict(population=population,**row))
    truths=gd.filters(d['retained'])['complete'];tq=[];owners=[]
    for j,(h,r,t) in enumerate(pq):
        for answer in sorted(truths.get((h,r),())):
            if answer!=t:owners.append((j,len(tq)));tq.append((h,r,answer))
    true=raw_ranks(model,tq,data,cfg);true.tofile(out/'retained_answers.u32le')
    np.asarray(owners,dtype='<u4').reshape(-1,2).tofile(out/'retained_answer_owners.u32le')
    if state!=rt.tensor_tree_hash(model.state_dict()):raise ValueError('Frozen inference changed model')
    summary=dict(seed=seed,phase=phase,method=cfg['name'],queries=len(queries),n_deleted=nd,metrics=stats,preferred=pref,
        retained_answer_count=len(tq),cohort='fixed final retained sample' if cfg['evaluation']=='sampled' else 'full current retained population',
        backbone=cfg.get('backbone','TransE'),preferred_scope='sampled current-request designated-side queries' if preferred_indices is not None else 'all current-request designated-side queries',
        preferred_request_indices=list(chosen),query_sha256=__import__('hashlib').sha256(ranks[:,:3].tobytes()).hexdigest())
    (out/'evaluation.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8');return summary

def run(cfg,data,seed,device,out,pretrained=None,smoke=False,calibration_only=False):
    if cfg.get('status')!='ready':raise ValueError('Reserved/unapproved configuration cannot run')
    if cfg.get('backbone','TransE') not in ('TransE','RotatE'):raise ValueError('Unsupported backbone')
    if cfg.get('backbone')=='RotatE' and cfg['evaluation']!='sampled':raise ValueError('RotatE requires separately labeled sampled evaluation')
    if cfg.get('processor')=='ADVIMP' and device.type!='cuda':raise ValueError('Original ADV-IMP requires CUDA; no replacement CPU algorithm is supplied')
    if cfg['batch_size']!=1024 or cfg['negative_ratio']!=10:raise ValueError('Pinned batch1024/negative10 required')
    if smoke and data['counts']['entities']>100:raise ValueError('Smoke overrides are restricted to a tiny artificial graph')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    if device.type=='cuda':torch.cuda.set_device(device)
    torch.set_num_threads(1 if smoke else 4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:
        cfg=copy.deepcopy(cfg)
        (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
        (out/'run.json').write_text(json.dumps(dict(seed=seed,dataset=data['dataset'],method=cfg['name'],backbone=cfg.get('backbone','TransE'),evaluation=cfg['evaluation'],smoke=smoke,torch=str(torch.__version__),device=str(device),calibration_only=calibration_only),indent=2)+'\n',encoding='utf-8')
        samples=None
        if cfg['evaluation']=='sampled':
            import query_order
            samples=query_order.build(data,data['dataset'],cfg.get('sampling'));(out/'samples.json').write_text(json.dumps(samples,indent=2)+'\n',encoding='utf-8')
        if pretrained:
            source=Path(pretrained);pin=json.loads((source.parent/'COMPLETE.json').read_text('utf-8'))['files'][source.name]
            if __import__('hashlib').sha256(source.read_bytes()).hexdigest()!=pin:raise ValueError('Shared pretraining checkpoint hash mismatch')
            p=torch.load(source,map_location='cpu',weights_only=True)
            if p['metadata']['kind']!='P' or p['metadata']['seed']!=seed:raise ValueError('Pretraining kind/seed mismatch')
            source_run=json.loads((source.parent/'run.json').read_text('utf-8'))
            if source_run.get('backbone','TransE')!=cfg.get('backbone','TransE') or source_run['dataset']!=data['dataset']:raise ValueError('Pretraining dataset or backbone mismatch')
        else:p=tr.pretrain(data,cfg,seed,device)
        rt.save_checkpoint_new(out/'pretrained.pt',p);prev=None;start=p
        for phase in range(1 if calibration_only else 4):
            folder=out/f'p{phase}';folder.mkdir()
            if cfg['processor']=='P':primary=p;meta={}
            elif cfg['processor']=='R':primary=tr.pretrain(data,cfg,seed,device,phase);meta={}
            else:
                deleted,meta=tr.delete(data,cfg,seed,phase,device,start,prev);prev=deleted
                rt.save_checkpoint_new(folder/'delete.pt',deleted)
                primary=tr.repair(data,cfg,seed,phase,device,deleted,tr.stage_repair_steps(cfg,len(gd.at(data,phase)['retained']))) if cfg['repair'] else deleted
                start=primary
            rt.save_checkpoint_new(folder/'model.pt',primary)
            pref=meta.pop('original_preferred',None) or original_preferred(data,cfg,seed,phase,device)
            (folder/'preferred_queries.json').write_text(json.dumps(pref)+'\n',encoding='utf-8')
            (folder/'training.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
            qs=None
            if samples is not None:
                selected=query_order.select(samples,data,phase);qs=(selected['queries'],selected['n_deleted'])
            pi=samples['deleted_by_phase'][phase]['request_indices'] if samples is not None and (cfg.get('backbone')=='RotatE' or data['dataset'] in ('CL','YA')) else None
            evaluate(primary,data,cfg,seed,phase,device,folder,pref,qs,pi)
            print(f'{cfg["name"]} seed={seed} request={phase+1} primary evaluated',flush=True)
            if phase==3 and cfg['diagnostic_probe'] and cfg['processor'] not in ('P',):
                steps=tr.diagnostic_steps(cfg,data,smoke)
                probe=tr.repair(data,cfg,seed,phase,device,primary,steps);po=folder/'probe';po.mkdir();rt.save_checkpoint_new(po/'model.pt',probe)
                evaluate(probe,data,cfg,seed,phase,device,po,pref,qs,pi)
        manifest={f.relative_to(out).as_posix():__import__('hashlib').sha256(f.read_bytes()).hexdigest() for f in out.rglob('*') if f.is_file()}
        (out/'COMPLETE.json').write_text(json.dumps(dict(files=manifest),indent=2)+'\n',encoding='utf-8')
        return out
    except BaseException:
        (out/'FAILED.txt').write_text(traceback.format_exc(),encoding='utf-8');raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--data',required=True);ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--device',default='cpu');ap.add_argument('--out',required=True);ap.add_argument('--pretrained');ap.add_argument('--evaluation',choices=('full','sampled'))
    ap.add_argument('--calibration-only',action='store_true')
    a=ap.parse_args();cfg=json.loads(Path(a.config).read_text('utf-8'))
    if a.evaluation:cfg['evaluation']=a.evaluation
    d=dat.load(a.data)
    if d['dataset'] in ('CL','YA') and cfg['evaluation']!='sampled':raise ValueError('CoDEx-L/YAGO3-10 use sampled final-retained evaluation; pass --evaluation sampled')
    if d['dataset'] in ('CL','YA'):cfg['query_batch_size']=4
    if a.calibration_only and not (d['dataset']=='WN' and a.seed==0 and cfg.get('backbone')=='RotatE' and cfg['name'] in ('IGPU-c2','IGPU-c3','IGPU-c4')):raise ValueError('Only fixed RotatE WN calibration candidates can run one request')
    run(cfg,d,a.seed,torch.device(a.device),a.out,a.pretrained,calibration_only=a.calibration_only)
    print('COMPLETE='+a.out)
if __name__=='__main__':main()
