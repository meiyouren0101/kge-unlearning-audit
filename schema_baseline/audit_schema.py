"""Import verified reference ranks into new audit views; use A v2 audit unchanged."""
import hashlib,os
from pathlib import Path
import numpy as np
from igpu import bootstrap,audit
import common as c,rank_kernels as rk
import matrix20 as m,schema_inputs as inp
from job20 import atomic,seal

def import_reference(data,key,arm,seed,k,out,a20):
    path,receipt=(inp.historical_source(key[:2],arm,seed,k) if key.endswith('10')
                  else inp.a_source(a20,key[:2],seed,k))
    raw=np.fromfile(path,dtype='<u4').reshape(-1,5);query,nd=inp.canonical(data,k)
    if receipt.get('query_sha256'):
        m.check(hashlib.sha256(raw[:,:3].tobytes()).hexdigest()==receipt['query_sha256'],'A query hash mismatch')
        m.check(receipt['n_deleted']==nd,'A retained/deleted boundary mismatch')
    a=inp.align(raw,query,nd,data['counts']['entities'])
    # Check source again before publishing the copied cache.
    m.verify_file(path,receipt['cache_sha256'])
    a.tofile(out/'ranks.u32le')
    c.write(out/'evaluation.json',dict(seed=seed,phase=k,method=arm,queries=len(a),n_deleted=nd,
        query_sha256=hashlib.sha256(a[:,:3].tobytes()).hexdigest(),metrics=rk.summaries(a[:,3:],nd),
        cohort='full current retained population',preferred=[]))
    c.write(out/'source_receipt.json',receipt)

def build(session,a20):
    plan=m.read(session/'plan.json');key=plan['key'];data=inp.graph(key,plan['archive'])
    views=session/'audit_views';views.mkdir();folders=[];coverage=[];resources=[];raw=[]
    for arm in ('P','R','Schema'):
        for seed in m.SEEDS:
            folder=views/f'{arm}_{seed}';folder.mkdir();folders.append(folder)
            c.write(folder/'run.json',dict(dataset=key[:2],method=arm,seed=seed,backbone='TransE',evaluation='full'))
            for k in range(4):
                dest=folder/f'p{k}';dest.mkdir()
                if arm=='R' or (arm=='P' and key.endswith('10')):
                    import_reference(data,key,arm,seed,k,dest,a20);status='COMPLETE'
                else:
                    stage=session/'stages'/f'evaluate_{arm}_{seed}_p{k}'
                    commit=m.read(stage/'commit.json');r=m.read(m.verify_file(stage/'result.json',commit['files']['result.json']));status=r['status']
                    if status=='COMPLETE':
                        for name in ('evaluation.json','ranks.u32le'):
                            source=m.verify_file(stage/name,commit['files'][name]);os.link(source,dest/name)
                coverage.append(dict(dataset=key[:2],deletion_ratio=int(key[2:]),method=arm,seed=seed,phase=k,status=status))
                if status=='COMPLETE':
                    for row in m.read(dest/'evaluation.json')['metrics']:raw.append(dict(dataset=key[:2],method=arm,seed=seed,phase=k,**row))
            c.write(folder/'COMPLETE.json',dict(files=m.signed_payload(folder)))
    audit.run(folders,data,session/'audit')
    c.csvwrite(session/'audit/coverage.csv',coverage);c.csvwrite(session/'audit/all_rank_metrics_by_seed.csv',raw)
    for spec in plan['stages']:
        r=m.read(session/'stages'/spec['name']/'result.json')
        resources.append({name:r.get(name) for name in ('name','arm','seed','phase','mode','status','reason','requested','updates','skipped','optimizer_steps','seconds','gpu_peak_reserved_mib')})
    c.csvwrite(session/'audit/stage_resources.csv',resources)
    all_ok=all(r['status']=='COMPLETE' for r in coverage)
    atomic(session/'complete.json',dict(status='COMPLETE' if all_ok else 'COMPLETE_WITH_UNAVAILABLE',
        audit_files=m.signed_payload(session/'audit'),views_sha256={str(p.relative_to(views)):m.digest(p) for p in views.glob('*/COMPLETE.json')}))
