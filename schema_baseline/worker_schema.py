"""One private Schema update or frozen evaluation; no R input in training."""
import argparse,time,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from worker20 import configure,start,evaluate,tr,rt,gd,torch,c,gpu,m,rb
import schema_adapter as schema
from schema_inputs import graph

def sklearn_contract():
    import sklearn
    from sklearn.cluster import KMeans
    p=KMeans(n_clusters=3,random_state=0).get_params()
    m.check(sklearn.__version__=='1.6.1','Use reviewed scikit-learn==1.6.1')
    m.check(p['n_init']=='auto' and p['init']=='k-means++' and p['algorithm']=='lloyd','KMeans defaults changed')
    return dict(sklearn=sklearn.__version__,KMeans=p,effective_n_init=1)

def update(data,cfg,source,out,seed,k,device):
    up,_=tr.model_api(cfg);a=tr.args_for(cfg,data,k,seed,device)
    model=up.TransE(a,gd.kg(data,k));rt.load_model_state(model,source['model']);model.train()
    for p in model.parameters():p.requires_grad_(True)
    rel=model.rel_embeddings.weight.detach().clone();rb.record('start',0,model)
    settings=sklearn_contract()
    row=schema.run_phase(model,data['triples'],data['phases'],k,seed,
        on_step=lambda i,m,o:rb.record('schema_update',i,m,o),
        on_progress=lambda i,n,u:print(f'SCHEMA_REQUEST={i}/{n}; UPDATES={u}',flush=True))
    m.check(torch.equal(rel,model.rel_embeddings.weight),'Schema unexpectedly changed relations')
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):raise FloatingPointError('Nonfinite Schema endpoint')
    m.check(row['requested']==row['updates']+row['skipped'],'Schema request accounting')
    losses=row.pop('losses');matches=row.pop('matches')
    row.update(settings,loss_sum=math.fsum(losses),match_trace_sha256=rt.tensor_tree_hash(matches),
               optimizer_steps=row['updates'],relation_weights_unchanged=True)
    end=dict(schema_version=1,model=rt.cpu_tree(model.state_dict()),optimizer=None,rng=rt.capture_rng(),
        metadata=dict(kind='P',phase=k,seed=seed,updates=row['updates']),loader_generator=None)
    rb.record('endpoint',row['updates'],model)
    return dict(row,checkpoint_sha256=rt.save_checkpoint_new(out/'final.pt',end))

def main():
    p=argparse.ArgumentParser();p.add_argument('session',type=Path);p.add_argument('stage');a=p.parse_args()
    session=a.session;plan=m.read(session/'plan.json');spec=next(x for x in plan['stages'] if x['name']==a.stage)
    out=session/'stages'/a.stage;began=time.monotonic();cfg=m.read(m.ROOT/'configs/P.json')
    try:
        torch.cuda.set_device(0);configure();torch.cuda.reset_peak_memory_stats()
        m.verify_file(plan['archive'],plan['archive_sha256']);data=graph(plan['key'],plan['archive'])
        seed,k=spec['seed'],spec['phase'];device=torch.device('cuda:0');rt.seed_rng(rt.schedule_seed(seed,k))
        with rb.Bridge(session,spec,out) as bridge:
            src,h=start(session,spec['tag'],seed,spec.get('source',spec.get('previous')));bridge.anchor('checkpoint',h)
            try:
                result=(update(data,cfg,src,out,seed,k,device) if spec['mode']=='delete' else
                    evaluate(data,cfg,seed,k,device,src,out,spec['arm'],extra_preferred=True))
                result.update(status='COMPLETE',source_checkpoint_sha256=h)
            except (FloatingPointError,ValueError) as exc:
                if not isinstance(exc,FloatingPointError) and 'Nonfinite frozen score' not in str(exc):raise
                result=dict(status='DIVERGED',reason=str(exc),source_checkpoint_sha256=h)
            bridge.finish()
        torch.cuda.synchronize()
        c.write(out/'result.json',dict(spec,**result,seconds=time.monotonic()-began,gpu_peak_reserved_mib=gpu.torch_peak_mib()))
    except BaseException as exc:
        if not gpu.is_cuda_oom(exc):raise
        try:snapshot=gpu.query()
        except Exception:snapshot=None
        gpu.oom_marker(out,spec,exc,snapshot=snapshot)
    finally:gpu.clear_stage_cache(out)
if __name__=='__main__':main()
