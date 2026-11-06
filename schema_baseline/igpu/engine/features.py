"""Current retained graph only; no model predictions or R/P labels are used."""
import math
import common as c
import io_utils as io
import graph_data as gd
def percentiles(v):
    import numpy as np
    x=np.asarray(v,dtype=np.float64);io.require(len(x)>0 and np.isfinite(x).all(),'Finite percentile population')
    _,ids,n=np.unique(x,return_inverse=True,return_counts=True);return ((np.cumsum(n)-n/2)/len(x))[ids]
def compute(data,phase):
    import numpy as np
    from proxy_core import RetainedFeatures
    d=gd.at(data,phase);queries=list(gd.queries(sorted(d['cumulative'])));extractor=RetainedFeatures(d['retained'])
    raw=np.asarray([extractor.row(q) for q in queries],dtype=np.float64);cols=[percentiles(raw[:,j]) for j in range(5)];cols[2]=1-cols[2]
    s3=np.column_stack(cols).mean(1);lookup={q:i for i,q in enumerate(queries)};current=list(gd.queries(data['phases'][phase]));scores=np.asarray([s3[lookup[q]] for q in current]);qs=np.asarray(current,dtype='<u4');q=np.empty(len(qs))
    for direction in (0,1):
        mask=qs[:,1]%2==direction;q[mask]=percentiles(scores[mask])
    w=2*(1-q);io.require(abs(float(w.mean())-1)<1e-14 and ((w>=0)&(w<=2)).all(),'G2 weights')
    return qs,raw,s3,q,w
def run(data,phase,out):
    import numpy as np
    qs,raw,s3,q,w=compute(data,phase)
    with (out/'weights.npz').open('xb') as f:np.savez(f,queries=qs,q=q,weights=w,S3_cumulative=s3)
    c.write(out/'receipt.json',dict(current_phase=phase,retained_facts=len(gd.at(data,phase)['retained']),cumulative_queries=len(raw),current_queries=len(qs),
        source='current retained graph only',labels_read=False,mean_weight=float(w.mean()),min_weight=float(w.min()),max_weight=float(w.max()),q_direction_blocks_separate=True))
    return dict(status='COMPLETE',mode='features',phase=phase,model_inference=False,training_updates=0)
