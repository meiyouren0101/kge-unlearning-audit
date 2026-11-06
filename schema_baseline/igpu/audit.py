"""CPU rank-cache audit: six ordered (a,b,c) roles and per-query absolute gaps."""
from . import bootstrap
from pathlib import Path
from collections import defaultdict
import argparse,hashlib,itertools,json,zipfile,math
import numpy as np
import common as c,absolute_metrics as am,audit_kernels as ak,graph_data as gd
from . import data as dat

def verify(folder):
    folder=Path(folder);seal=json.loads((folder/'COMPLETE.json').read_text('utf-8'))
    for name,h in seal['files'].items():
        if name.endswith(('.u32le','.json')) and hashlib.sha256((folder/name).read_bytes()).hexdigest()!=h:raise ValueError('Changed audit input '+name)
    return json.loads((folder/'run.json').read_text('utf-8'))

def cache(folder,phase,endpoint):
    path=Path(folder)/f'p{phase}'
    if endpoint=='probe':path=path/'probe'
    arr=np.fromfile(path/'ranks.u32le',dtype='<u4').reshape(-1,5)
    meta=json.loads((path/'evaluation.json').read_text('utf-8'));order=np.lexsort((arr[:,2],arr[:,1],arr[:,0]))
    keys=arr[order,:3];deleted=order<meta['n_deleted']
    if len(keys)>1 and not np.any(keys[1:]!=keys[:-1],axis=1).all():raise ValueError('Duplicate cached query')
    return keys,deleted,arr[order,4].astype(float),meta

def safe_mean(block,keys):
    out={}
    for k in keys:
        v=[r.get(k) for r in block];out[k]=math.fsum(v)/len(v) if all(x is not None for x in v) else None
    return out

def run(folders,data,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);registry={};task=[];preferred=[];backbones=set();scopes=set()
    for folder in map(Path,folders):
        meta=verify(folder)
        if meta['dataset']!=data['dataset']:raise ValueError('Cannot pool different graphs')
        backbones.add(meta.get('backbone','TransE'));scopes.add(meta.get('evaluation','full'))
        if len(backbones)>1 or len(scopes)>1:raise ValueError('Do not pool backbones or sampled/full evaluation scopes')
        key=(meta['method'],meta['seed'])
        if key in registry:raise ValueError('Duplicate method/seed')
        registry[key]=folder
        for phase in range(4):
            for endpoint in ('primary','probe'):
                path=folder/f'p{phase}';path=path/'probe' if endpoint=='probe' else path
                if not (path/'evaluation.json').is_file():continue
                e=json.loads((path/'evaluation.json').read_text('utf-8'));vals={(x['filter'],x['cohort'],x['direction']):x for x in e['metrics']}
                for filt,direction in itertools.product(('complete','legacy'),('combined','forward','inverse')):
                    ret=vals[filt,'all_retained',direction];deleted=vals[filt,'cumulative_forget',direction];avg,f1=ak.paper_scores(ret['MRR'],deleted['MRR'])
                    task.append(dict(method=meta['method'],seed=meta['seed'],phase=phase,endpoint=endpoint,filter=filt,direction=direction,
                        retained_MRR=ret['MRR'],deleted_MRR=deleted['MRR'],MRR_Avg=avg,MRR_F1=f1,cohort=e['cohort']))
                for r in e['preferred']:preferred.append(dict(method=meta['method'],seed=meta['seed'],phase=phase,endpoint=endpoint,**r))
    c.csvwrite(out/'task_metrics_by_seed.csv',task);c.csvwrite(out/'preferred_by_seed.csv',preferred)
    for label,items,keys,fields in (('task_metrics_mean',task,('method','phase','endpoint','filter','direction'),('retained_MRR','deleted_MRR','MRR_Avg','MRR_F1')),
        ('preferred_mean',preferred,('method','phase','endpoint','population','direction'),('MRR','Hit10'))):
        grouped=defaultdict(list)
        for r in items:grouped[tuple(r[k] for k in keys)].append(r)
        mean=[]
        for key,block in grouped.items():
            complete={r['seed'] for r in block}=={0,42,123}
            mean.append(dict(zip(keys,key),seeds_observed=len(block),**{k:math.fsum(r[k] for r in block)/len(block) if complete and all(r[k] is not None for r in block) else None for k in fields}))
        c.csvwrite(out/(label+'.csv'),mean)
    seeds=(0,42,123)
    required=[(a,s) for a in ('R','P') for s in seeds]
    missing=[x for x in required if x not in registry]
    if missing:
        (out/'AUDIT_STATUS.json').write_text(json.dumps(dict(status='MISSING_REFERENCES',missing=missing,
            note='Six-role audit requires three independent R and P seed blocks; only raw task metrics emitted.'),indent=2)+'\n');return
    methods=sorted({m for m,s in registry if m not in ('P','R')})+['P','R_control']
    rows=[];means=[];recovery=[]
    for phase in range(4):
        refs={};pres={};keys=None;dm=None
        def read(folder,endpoint='primary'):
            nonlocal keys,dm
            q,d,r,e=cache(folder,phase,endpoint)
            if keys is None:keys=q;dm=d
            elif not np.array_equal(keys,q) or not np.array_equal(dm,d):raise ValueError('Exact query/cohort identity mismatch')
            return r
        for s in seeds:refs[s]=read(registry['R',s]);pres[s]=read(registry['P',s])
        byfact={q:k for k,qs in enumerate(data['phases']) for q in qs}
        phases=np.array([byfact.get((int(t),int(r)-1,int(h)) if r%2 else (int(h),int(r),int(t)),-1) for h,r,t in keys])
        for method in methods:
            for endpoint in ('primary','probe'):
                if endpoint=='probe' and (phase!=3 or method in ('P','R_control')):continue
                if method not in ('P','R_control') and any((method,s) not in registry or not (registry[method,s]/f'p{phase}/evaluation.json').is_file() for s in seeds):continue
                if endpoint=='probe' and any(not (registry[method,s]/f'p{phase}/probe/evaluation.json').is_file() for s in seeds):continue
                xs={s:pres[s] if method=='P' else read(registry[method,s],endpoint) for s in seeds} if method!='R_control' else {}
                cells=defaultdict(list);arrays=defaultdict(list)
                for a,b,control in itertools.permutations(seeds,3):
                    x=refs[control] if method=='R_control' else xs[b];ra=refs[a];rb=refs[b];p=pres[b]
                    for cohort,direction,binname,source in itertools.product(('cumulative_forget','all_retained'),('combined','forward','inverse'),
                        ('all','R_le10','R_11to100','R_gt100','focus_Pb_le10_Ra_gt100'),('all',0,1,2,3)):
                        if cohort=='all_retained' and source!='all':continue
                        mask=dm.copy() if cohort=='cumulative_forget' else ~dm
                        if direction!='combined':mask&=keys[:,1]%2==int(direction=='inverse')
                        if source!='all':mask&=phases==source
                        if binname=='R_le10':mask&=ra<=10
                        elif binname=='R_11to100':mask&=(ra>10)&(ra<=100)
                        elif binname=='R_gt100':mask&=ra>100
                        elif binname=='focus_Pb_le10_Ra_gt100':mask&=(p<=10)&(ra>100)
                        value,absvalues=am.summarize(x,rb,mask);old=ak.stat(x,rb,p,mask)
                        meta=dict(method=method,phase=phase,endpoint=endpoint,cohort=cohort,direction=direction,R_bin=binname,deletion_phase=source)
                        row={**meta,'bucket_seed_a':a,'reference_and_arm_seed_b':b,'control_seed_c':control,**value,**old}
                        rows.append(row);key=(cohort,direction,binname,source);cells[key].append(row);arrays[key].append(absvalues)
                for key,block in cells.items():
                    means.append(dict(method=method,phase=phase,endpoint=endpoint,cohort=key[0],direction=key[1],R_bin=key[2],deletion_phase=key[3],
                        **am.aggregate(block,arrays[key]),**safe_mean(block,('signed','abs','DeltaMRR','X_minus_P'))))
    c.csvwrite(out/'cross_bins_by_combination.csv',rows);c.csvwrite(out/'cross_bins_mean.csv',means)
    seedrows=[]
    groups=defaultdict(list)
    for row in rows:
        key=tuple(row[k] for k in ('method','phase','endpoint','cohort','direction','R_bin','deletion_phase','reference_and_arm_seed_b'));groups[key].append(row)
    for key,block in groups.items():seedrows.append(dict(zip(('method','phase','endpoint','cohort','direction','R_bin','deletion_phase','seed'),key),
        **safe_mean(block,('signed','abs','DeltaMRR','X_minus_P','abs_log_mean','positive_log_mean','negative_log_mean'))))
    c.csvwrite(out/'cross_bins_by_seed.csv',seedrows)
    primary={(r['method'],r['cohort'],r['direction'],r['R_bin'],r['deletion_phase']):r for r in means if r['phase']==3 and r['endpoint']=='primary'}
    for row in means:
        if row['endpoint']!='probe':continue
        before=primary[row['method'],row['cohort'],row['direction'],row['R_bin'],row['deletion_phase']]
        recovery.append(dict(method=row['method'],cohort=row['cohort'],direction=row['direction'],R_bin=row['R_bin'],deletion_phase=row['deletion_phase'],
            endpoint_signed=before['signed'],probe_signed=row['signed'],recovery=None if row['signed'] is None or before['signed'] is None else before['signed']-row['signed']))
    c.csvwrite(out/'repair_recovery.csv',recovery)
    status=dict(status='COMPLETE',roles=6,seeds=list(seeds),backbone=next(iter(backbones)),evaluation=next(iter(scopes)),
        retained_scope='sampled fixed final-retained queries; not full current-retained main-table population' if 'sampled' in scopes else 'full current-retained population',
        R_control='R_c minus R_b; buckets from independent R_a',
        definitions='Positive log(rank_X)-log(rank_Rb): weaker retrieval than retraining; negative: stronger. Absolute computed per query before averaging.',
        non_independence='Facts/queries, reciprocal directions, six roles and seeds on one graph are not independent datasets.')
    (out/'AUDIT_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    report=dict(status,task_rows=len(task),crossfit_rows=len(rows),mean_rows=len(means));(out/'report.txt').write_text(json.dumps(report,indent=2)+'\n')
    with zipfile.ZipFile(out/'audit_report.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in out.iterdir():
            if f.suffix in ('.csv','.json','.txt'):z.write(f,f.name)

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    run(a.runs,dat.load(a.data),a.out)
if __name__=='__main__':main()
