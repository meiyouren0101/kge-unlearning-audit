import math

def paper_scores(ret,deleted):
    if ret is None or deleted is None:return None,None
    u=1-deleted;den=ret+u;return den/2,2*ret*u/den if den else None

def stat(x,ref,p,mask):
    import numpy as np
    n=int(mask.sum())
    if not n or x is None or ref is None or p is None:return dict(query_count=n,signed=None,abs=None,DeltaMRR=None,X_minus_P=None)
    gap=np.log(x[mask])-np.log(ref[mask]);return dict(query_count=n,signed=float(gap.mean()),abs=float(np.abs(gap).mean()),DeltaMRR=float((1/x[mask]-1/ref[mask]).mean()),X_minus_P=float((np.log(x[mask])-np.log(p[mask])).mean()))
