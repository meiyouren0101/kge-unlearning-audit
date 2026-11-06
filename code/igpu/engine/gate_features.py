import hashlib,math,random
from collections import Counter
import numpy as np
import io_utils as io
import features
def degree_scores(retained,queries):
    degree=Counter()
    for h,r,t in retained:
        degree[h]+=1
        if h!=t:degree[t]+=1
    return np.asarray([degree[int(t)] for h,r,t in queries],dtype=np.float64)
def permutation(n,phase,direction,identity=False):
    ids=list(range(n));seed=int.from_bytes(hashlib.sha256(f'graphdpo-gate-shuffle-v1/20261031/WN/{phase}/{direction}'.encode()).digest()[:8],'little')
    if not identity:random.Random(seed).shuffle(ids)
    return np.asarray(ids,dtype=np.int64),seed
def variants(retained,queries,original_w,phase,identity=False,multiplier=2.):
    scores=degree_scores(retained,queries);degree=np.empty(len(queries));degree_q=np.empty(len(queries));shuffled=original_w.copy();perm=np.arange(len(queries),dtype=np.int64);seeds={}
    for direction in (0,1):
        ids=np.flatnonzero(queries[:,1]%2==direction);degree_q[ids]=features.percentiles(scores[ids]);degree[ids]=2*(1-degree_q[ids]);sub,seed=permutation(len(ids),phase,direction,identity);shuffled[ids]=original_w[ids[sub]];perm[ids]=ids[sub];seeds[str(direction)]=seed
        io.require(np.array_equal(np.sort(shuffled[ids]),np.sort(original_w[ids])) and math.fsum(map(float,shuffled[ids]))==math.fsum(map(float,original_w[ids])),'Shuffled distribution changed')
    return {'degree':degree,'shuffled':shuffled,'c20':original_w*multiplier},dict(permutation=perm,stream_seeds=seeds,raw_degree=scores,degree_q=degree_q)
