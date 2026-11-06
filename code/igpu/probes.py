"""Fixed matched alternative-answer probe, after the selected histories finish."""
from . import bootstrap
from pathlib import Path
from collections import defaultdict
import argparse,json,math
import numpy as np
import torch
from . import audit,data as dat,training as tr,run
import common as c,graph_data as gd,probe_kernels as pk
def read_payload(folder,name):
    import hashlib
    path=Path(folder)/name;pin=json.loads((Path(folder)/'COMPLETE.json').read_text('utf-8'))['files'][name]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=pin:raise ValueError('Checkpoint changed before probe')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=pin:raise ValueError('Checkpoint changed during probe')
    return payload

def execute(folders,data,out,limit=2048,device='cpu'):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);registry={};truths_all=set(gd.queries(data['triples']));forbidden=set(truths_all)
    for f in map(Path,folders):
        md=audit.verify(f);registry[md['method'],md['seed']]=f
        for phase in range(4):
            t=json.loads((f/f'p{phase}/training.json').read_text('utf-8'));actual=t.get('actual_positives') or []
            forbidden.update(gd.queries(actual))
    allrows=[];truthrows=[]
    for seed in sorted({s for m,s in registry}):
        seed_runs={m:f for (m,s),f in registry.items() if s==seed}
        for phase in range(4):
            first=next(iter(seed_runs.values()));original=json.loads((first/f'p{phase}/preferred_queries.json').read_text('utf-8'))
            for f in seed_runs.values():
                if json.loads((f/f'p{phase}/preferred_queries.json').read_text('utf-8'))!=original:raise ValueError('Original preferred queries differ across methods')
            plan=pk.build_one(data,phase,seed,original,forbidden,limit);po=out/f's{seed}_p{phase}';po.mkdir()
            np.savez(po/'plan.npz',**{k:plan[k] for k in ('independent','actual','truth','truth_owners')});c.csvwrite(po/'positions.csv',plan['rows'])
            for method,f in seed_runs.items():
                cfg=json.loads((f/'config.json').read_text('utf-8'));payload=read_payload(f,f'p{phase}/model.pt')
                model=tr.load_model(data,cfg,seed,phase,torch.device(device),payload);ranks={};dest=po/method;dest.mkdir()
                for label in ('actual','independent','truth'):
                    qs=[tuple(q) for q in plan[label].tolist()];r=run.raw_ranks(model,qs,data,cfg,batch_size=16);r.tofile(dest/(label+'.u32le'));ranks[label]=r
                unobserved=np.array([tuple(q) not in truths_all for q in plan['actual']])
                for label,mask in (('all_original_preferred',np.ones(len(plan['actual']),bool)),('unobserved_original_preferred',unobserved),('independent_unobserved',np.ones(len(plan['independent']),bool))):
                    arr=ranks['independent' if label=='independent_unobserved' else 'actual'][mask]
                    for row in pk.metrics(arr[:,4],arr[:,:3]):allrows.append(dict(method=method,seed=seed,phase=phase,population=label,**row))
                owners=plan['truth_owners'];rankvals=ranks['truth'][:,4]
                summaries=[dict(weighting='unique_query',**row) for row in pk.metrics(rankvals,plan['truth'])]
                paired=rankvals[owners[:,1]] if len(owners) else np.array([])
                pqs=plan['independent'][owners[:,0]] if len(owners) else np.empty((0,3),dtype='<u4')
                summaries += [dict(weighting='request_answer_pair',**row) for row in pk.metrics(paired,pqs)]
                summaries += [dict(weighting='request_mean_over_true_answers',**row) for row in pk.truth_request_metrics(rankvals,plan)]
                source='pretrained.pt' if phase==0 else f'p{phase-1}/model.pt'
                old=read_payload(f,source);oldm=tr.load_model(data,cfg,seed,phase,torch.device(device),old)
                oldr=run.raw_ranks(oldm,[tuple(q) for q in plan['truth'].tolist()],data,cfg,batch_size=16)
                for row in summaries:
                    ids=np.ones(len(rankvals),bool) if row['direction']=='combined' else plan['truth'][:,1]%2==int(row['direction']=='inverse')
                    row['DeltaMRR_endpoint_minus_stage_input']=float((1/rankvals[ids]-1/oldr[ids,4]).mean()) if ids.any() and row['weighting']=='unique_query' else None
                    truthrows.append(dict(method=method,seed=seed,phase=phase,**row))
    for row in truthrows:
        p=next((x for x in truthrows if x['method']=='P' and x['seed']==row['seed'] and x['phase']==row['phase'] and x['direction']==row['direction'] and x['weighting']==row['weighting']),None)
        row['DeltaMRR_vs_P']=row['MRR']-p['MRR'] if p and p['MRR'] is not None and row['MRR'] is not None else None
    c.csvwrite(out/'preferred_probe_by_seed.csv',allrows);c.csvwrite(out/'true_answers_by_seed.csv',truthrows)
    for name,rows,keys,fields in (('preferred_probe_mean',allrows,('method','phase','population','direction'),('MRR','Hit10')),
        ('true_answers_mean',truthrows,('method','phase','direction','weighting'),('MRR','DeltaMRR_vs_P','DeltaMRR_endpoint_minus_stage_input'))):
        groups=defaultdict(list)
        for r in rows:groups[tuple(r[k] for k in keys)].append(r)
        result=[]
        for key,block in groups.items():
            complete={r['seed'] for r in block}=={0,42,123}
            result.append(dict(zip(keys,key),seeds=len(block),**{field:math.fsum(r[field] for r in block)/len(block) if complete and all(r[field] is not None for r in block) else None for field in fields}))
        c.csvwrite(out/(name+'.csv'),result)
    (out/'PROBE_STATUS.json').write_text(json.dumps(dict(status='COMPLETE',limit=limit,seed_rule='original frozen fair-probe hash streams',
        scope='common fixed probe subset, at most2048 requests per seed/phase; not all preferred',
        excludes='known truths and selected histories actual positive targets; training negatives are not excluded',
        warning='unobserved is not a false-fact label; no rank-displacement mechanism claim',device=device),indent=2)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
    torch.set_num_threads(4);execute(a.runs,dat.load(a.data),a.out,device=a.device)
if __name__=='__main__':main()
