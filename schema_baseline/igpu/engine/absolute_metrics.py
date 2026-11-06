"""Absolute and one-sided query-level errors; signs never cancel before abs."""
import math

FIELDS=('abs_log_mean','abs_log_median','positive_log_mean','positive_fraction','positive_log_mean_conditional',
        'negative_log_mean','negative_fraction','negative_log_mean_conditional','equal_fraction','signed_log_mean_secondary',
        'MRR_X','MRR_Rb','MRR_damage_per_query','MRR_drop_of_means_clipped')

def summarize(x,ref,mask):
    import numpy as np
    n=int(mask.sum())
    if x is None or ref is None or not n:return dict(query_count=n,**{k:None for k in FIELDS}),None
    a=x[mask];b=ref[mask];g=np.log(a)-np.log(b);positive=a>b;negative=a<b
    absolute=np.sort(np.abs(g));pos=np.maximum(g,0);neg=np.maximum(-g,0);rx=1/a;rr=1/b
    return dict(query_count=n,abs_log_mean=float(absolute.mean()),abs_log_median=float(np.median(absolute)),
        positive_log_mean=float(pos.mean()),positive_fraction=float(positive.mean()),positive_log_mean_conditional=float(g[positive].mean()) if positive.any() else None,
        negative_log_mean=float(neg.mean()),negative_fraction=float(negative.mean()),negative_log_mean_conditional=float(-g[negative].mean()) if negative.any() else None,
        equal_fraction=float((a==b).mean()),signed_log_mean_secondary=float(g.mean()),MRR_X=float(rx.mean()),MRR_Rb=float(rr.mean()),
        MRR_damage_per_query=float(np.maximum(rr-rx,0).mean()),MRR_drop_of_means_clipped=max(float(rr.mean()-rx.mean()),0.)),absolute

def mixture_median(groups):
    """Exact median of equal-mass empirical groups, not mean(group medians).

    Sorted discrete-support selection avoids a huge concatenated weighted sort.
    Equal counts reduce exactly to the ordinary concatenated-sample median.
    """
    import numpy as np
    if not groups or any(v is None or not len(v) for v in groups):return None
    k=len(groups);lo=[0]*k;hi=[len(v) for v in groups];answer=None;eps=1e-14
    def cdf(v):return math.fsum(int(np.searchsorted(a,v,side='right'))/len(a) for a in groups)/k
    while any(a<b for a,b in zip(lo,hi)):
        candidates=sorted(float(a[(l+h)//2]) for a,l,h in zip(groups,lo,hi) if l<h);pivot=candidates[len(candidates)//2]
        if cdf(pivot)>=.5-eps:
            answer=pivot;hi=[min(h,int(np.searchsorted(a,pivot,side='left'))) for a,h in zip(groups,hi)]
        else:lo=[max(l,int(np.searchsorted(a,pivot,side='right'))) for a,l in zip(groups,lo)]
    assert answer is not None
    if abs(cdf(answer)-.5)<=eps:
        following=[]
        for a in groups:
            i=int(np.searchsorted(a,answer,side='right'))
            if i<len(a):following.append(float(a[i]))
        if following:return (answer+min(following))/2
    return answer

def aggregate(rows,absolute_groups):
    out=dict(planned_roles=len(rows),defined_roles=sum(r['abs_log_mean'] is not None for r in rows),
             role_query_observations=sum(r['query_count'] for r in rows),query_count_mean=math.fsum(r['query_count'] for r in rows)/len(rows))
    # Empty/missing roles remain undefined, never padded with zero.
    for key in FIELDS:
        values=[r[key] for r in rows if r[key] is not None]
        out[key]=math.fsum(values)/len(rows) if len(values)==len(rows) else None
    out['abs_log_median']=mixture_median(absolute_groups)
    out['MRR_drop_of_means_clipped']=max(out['MRR_Rb']-out['MRR_X'],0.) if out['MRR_X'] is not None and out['MRR_Rb'] is not None else None
    # Conditional means pool equal-mass role positive/negative distributions.
    for side in ('positive','negative'):
        m=out[side+'_log_mean'];q=out[side+'_fraction']
        out[side+'_log_mean_conditional']=m/q if m is not None and q is not None and q>0 else None
    return out
