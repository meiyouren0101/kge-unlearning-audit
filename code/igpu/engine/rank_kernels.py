"""Verbatim complete/legacy filtering and task metrics."""
import io_utils as io

def kernel(torch,raw,mask,target):
    ids=torch.arange(len(raw),device=raw.device);keep=raw[ids,target].clone();values=torch.where(mask,-torch.ones_like(raw)*10000000,raw);values[ids,target]=keep
    return 1+torch.argsort(torch.argsort(values,dim=1,descending=True),dim=1,descending=False)[ids,target]

def score(torch,model,qs,index,device,filters=('complete','legacy')):
    x=torch.tensor(qs,dtype=torch.long,device=device);raw=model.predict(x[:,0],x[:,1])
    io.require(bool(torch.isfinite(raw).all()),'Nonfinite frozen score');out={}
    for filt in filters:
        mask=torch.zeros_like(raw,dtype=torch.bool)
        for i,(h,r,t) in enumerate(qs):mask[i,sorted(index[filt].get((h,r),()))]=True
        out[filt]=kernel(torch,raw,mask,x[:,2]).cpu().tolist()
    return out

def summaries(records,n_deleted):
    import numpy as np
    x=np.asarray(records,dtype=np.float64);rows=[]
    for filt,col in (('legacy',0),('complete',1)):
        for cohort,v in (('cumulative_forget',x[:n_deleted,col]),('all_retained',x[n_deleted:,col])):
            for direction in ('combined','forward','inverse'):
                a=v if direction=='combined' else v[int(direction=='inverse')::2]
                rows.append(dict(filter=filt,cohort=cohort,direction=direction,count=len(a),MR=float(a.mean()),MRR=float((1/a).mean()),**{'Hit'+str(k):float((a<=k).mean()) for k in (1,3,10)}))
    return rows
