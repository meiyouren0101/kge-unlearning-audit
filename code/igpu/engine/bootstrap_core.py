"""Ordinary paired fact bootstrap; every direction/seed/role shares the same weights."""
from pathlib import Path
import hashlib,math
import numpy as np
import io_utils as io

class Columns:
    def __init__(self,folder,n):
        self.path=Path(folder)/'bootstrap_scratch.f64';self.n=n;self.handle=self.path.open('xb');self.indices={};self.count=0
    def add(self,key,fids,values):
        if key in self.indices:return self.indices[key]
        fids=np.asarray(fids,dtype=np.int64);values=np.asarray(values,dtype=float)
        io.require(len(fids)==len(values) and bool(np.isfinite(values).all()),'Invalid sufficient statistics')
        out=np.bincount(fids,weights=values,minlength=self.n)
        io.require(len(out)==self.n,'Fact ID out of range')
        self.handle.write(out.astype('<f8').tobytes());self.indices[key]=self.count;self.count+=1;return self.count-1
    def close(self):
        self.handle.flush();self.handle.close()
        return np.memmap(self.path,mode='r',dtype='<f8',shape=(self.count,self.n))

def fact_weights(rng,n):
    return np.bincount(rng.integers(0,n,size=n),minlength=n).astype('<u4')

def percentile(values):
    values=np.asarray(values,dtype=float);valid=np.isfinite(values)
    if not bool(valid.all()):return None,None,int(valid.sum())
    a,b=np.percentile(values,[2.5,97.5],method='linear');return float(a),float(b),len(values)

def resample(matrix,groups,replicates,seed,progress=None,batch_size=8):
    n=matrix.shape[1];rng=np.random.default_rng(seed);digest=hashlib.sha256()
    out=np.full((replicates,len(groups)),np.nan)
    for lo in range(0,replicates,batch_size):
        count=min(batch_size,replicates-lo)
        raw=np.stack([fact_weights(rng,n) for _ in range(count)]);digest.update(raw.tobytes())
        totals=raw.astype(float)@matrix.T
        for j,g in enumerate(groups):
            if g['point'] is None:continue
            diffs=[]
            for pair in g['components']:
                an,ad,bn,bd=pair
                with np.errstate(divide='ignore',invalid='ignore'):
                    diffs.append(totals[:,an]/totals[:,ad]-totals[:,bn]/totals[:,bd])
            # Exactly the requested order: contrasts within each role, six-role mean.
            out[lo:lo+count,j]=np.stack(diffs).mean(axis=0)
        if progress:progress(lo+count,replicates)
    return out,digest.hexdigest()

def fractional_truth_edges(plan,anchor_fact_ids):
    """Preserve old unique-query estimand while allocating shared queries to owners.

    A truth query has weight = mean(weights of distinct owning deletion facts).
    Thus w=1 reproduces the old unique-query mean without counting an answer twice.
    """
    owners=np.asarray(plan['truth_owners'],dtype=np.int64)
    if not len(owners):return np.array([],int),np.array([],int),np.array([],float)
    pairs=np.unique(np.column_stack((np.asarray(anchor_fact_ids)[owners[:,0]],owners[:,1])),axis=0)
    counts=np.bincount(pairs[:,1],minlength=len(plan['truth']))
    io.require(bool((counts>0).all()),'Truth query without an owning request')
    return pairs[:,0],pairs[:,1],1/counts[pairs[:,1]]
