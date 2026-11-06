"""Fixed directed AUC, independent-R reference and paired fact-cluster bootstrap."""
import hashlib,itertools,math
import io_utils as io
METRICS=('AUC','AUC_R','AUC_minus_R','aux_AUC','aux_abs_from_half','RR_primary_noise','RR_primary_abs_noise',
         'RR_aux_AUC','RR_aux_abs_from_half','Ra_selected_primary_noise_secondary','Ra_selected_aux_AUC_secondary')

def stable_seed(*items):return int.from_bytes(hashlib.sha256('/'.join(map(str,items)).encode()).digest()[:8],'little')

def auc(positive,negative,pw=None,nw=None):
    import numpy as np
    p=np.asarray(positive,dtype=int);n=np.asarray(negative,dtype=int)
    if not len(p) or not len(n):return None
    pw=np.ones(len(p)) if pw is None else np.asarray(pw,dtype=float);nw=np.ones(len(n)) if nw is None else np.asarray(nw,dtype=float)
    io.require(bool((p>=1).all()) and bool((n>=1).all()) and bool((pw>=0).all()) and bool((nw>=0).all()),'Invalid rank/weight')
    a=float(pw.sum());b=float(nw.sum())
    if not a or not b:return None
    hist=np.bincount(n,weights=nw,minlength=max(int(p.max()),int(n.max()))+1)
    return float(np.dot(pw,(np.cumsum(hist)-.5*hist)[p])/(a*b))

def mask(ranks,stratum,direction):
    import numpy as np
    r=np.asarray(ranks).reshape(-1);q=np.arange(len(r))
    v={'all':np.ones(len(r),dtype=bool),'R_le10':r<=10,'R_11to100':(r>10)&(r<=100),'R_gt100':r>100}[stratum]
    if direction!='combined':v&=q%2==int(direction=='inverse')
    return np.flatnonzero(v)

def weights(nd,nr,cfg,g,p,out):
    import numpy as np
    w=np.lib.format.open_memmap(out/'fact_cluster_weights.npy',mode='w+',dtype='<u2',shape=(cfg['bootstrap_replicates'],nd+nr))
    rng=np.random.default_rng(stable_seed(cfg['bootstrap_seed'],g,p))
    for i in range(len(w)):
        for off,n in ((0,nd),(nd,nr)):
            h=np.bincount(rng.integers(0,n,size=n),minlength=n);io.require(int(h.max())<65536,'Bootstrap weight overflow');w[i,off:off+n]=h
    w.flush();return w

def two_auc(d,r,reference_d,ids,w,nd,cluster):
    """Primary D-vs-matched-R and auxiliary X-vs-Rb on same D facts."""
    import numpy as np
    x=np.asarray(d).reshape(-1)[ids];y=np.asarray(r).reshape(-1)[ids];ref=np.asarray(reference_d).reshape(-1)[ids]
    values=np.full((len(w),2),np.nan);point=(auc(x,y),auc(x,ref))
    if point[0] is None:return point,values
    fids=ids//2;cids=nd+cluster[fids]
    for j in range(len(w)):
        pw=w[j,fids].astype(float);nw=pw*w[j,cids].astype(float)
        a=auc(x,y,pw,nw);b=auc(x,ref,pw,pw)
        if a is not None:values[j,0]=a
        if b is not None:values[j,1]=b
    return point,values

def ci(values,total):
    import numpy as np
    v=np.asarray(values);v=v[np.isfinite(v)];n=len(v)
    if n<math.ceil(.95*total):return None,None,n
    a,b=np.quantile(v,[.025,.975]);return float(a),float(b),n

def intervals(row,rep,metrics=METRICS):
    for j,k in enumerate(metrics):
        lo,hi,n=ci(rep[:,j],len(rep));row[k+'_CI95_low']=lo;row[k+'_CI95_high']=hi;row[k+'_bootstrap_defined']=n
    return row

def measures(point,rep,base,baserep,cpoint,crep,apoint,arep):
    import numpy as np
    if point[0] is None or base[0] is None:return {k:None for k in METRICS},np.full((len(rep),len(METRICS)),np.nan)
    v=(point[0],base[0],point[0]-base[0],point[1],abs(point[1]-.5),cpoint[0]-base[0],abs(cpoint[0]-base[0]),cpoint[1],abs(cpoint[1]-.5),apoint[0]-base[0],apoint[1])
    arr=np.column_stack((rep[:,0],baserep[:,0],rep[:,0]-baserep[:,0],rep[:,1],abs(rep[:,1]-.5),crep[:,0]-baserep[:,0],abs(crep[:,0]-baserep[:,0]),crep[:,1],abs(crep[:,1]-.5),arep[:,0]-baserep[:,0],arep[:,1]))
    return dict(zip(METRICS,v)),arr

def aggregate(rows,reps,keys,expected,metrics=METRICS):
    import numpy as np
    groups={};out=[]
    for r,v in zip(rows,reps):groups.setdefault(tuple(r[k] for k in keys),[]).append((r,v))
    for key,items in groups.items():
        complete=len(items)==expected and all(r[metrics[0]] is not None for r,_ in items)
        row=dict(zip(keys,key));row.update(planned_roles=expected,available_roles=len(items),defined_roles=sum(r[metrics[0]] is not None for r,_ in items),
           seeds_observed=len({r['seed'] for r,_ in items}),complete_planned_mean=complete,query_count_role_mean=float(np.mean([r['query_count'] for r,_ in items])))
        for k in metrics:
            v=[r[k] for r,_ in items if r[k] is not None];row[k]=float(np.mean(v)) if complete else None
            row[k+'_available_roles_mean']=float(np.mean(v)) if v else None
        rep=np.mean(np.stack([v for _,v in items]),axis=0) if complete else np.full_like(items[0][1],np.nan)
        out.append(intervals(row,rep,metrics))
    return out

def phase(graph,ph,values,available,loader,matched,cluster,receipt,cfg,out,notify):
    nd=int(loader.deleted.sum());nr=int(max(cluster))+1;w=weights(nd,nr,cfg,graph,ph,out)
    try:return phase_with_weights(graph,ph,values,available,loader,matched,cluster,receipt,cfg,out,notify,w)
    finally:w._mmap.close()

def phase_with_weights(graph,ph,values,available,loader,matched,cluster,receipt,cfg,out,notify,w):
    import numpy as np
    nd=int(loader.deleted.sum())
    d={k:v[loader.deleted] for k,v in values.items()};r={k:v[~loader.deleted][matched] for k,v in values.items()}
    baseline={};roles=[];reps=[]
    for a,b,c in itertools.permutations(cfg['seeds']):
        for s in cfg['strata']:
            for direction in cfg['directions']:
                ids=mask(d['R',a],s,direction)
                bb,bv=two_auc(d['R',b],r['R',b],d['R',b],ids,w,nd,cluster)
                cc,cv=two_auc(d['R',c],r['R',c],d['R',b],ids,w,nd,cluster)
                aa,av=two_auc(d['R',a],r['R',a],d['R',b],ids,w,nd,cluster)
                baseline[a,b,s,direction]=(ids,bb,bv,cc,cv,aa,av)
        notify('R_noise',a,b)
    roster=['P','R',*cfg['baseline_arms'],*cfg['method_arms']]
    for arm in roster:
        for a,b,c in itertools.permutations(cfg['seeds']):
            if (arm,b) not in available:continue
            for s in cfg['strata']:
                for direction in cfg['directions']:
                    ids,bb,bv,cc,cv,aa,av=baseline[a,b,s,direction]
                    if arm=='R':pp,pv=bb,bv
                    else:pp,pv=two_auc(d[arm,b],r[arm,b],d['R',b],ids,w,nd,cluster)
                    stats,v=measures(pp,pv,bb,bv,cc,cv,aa,av)
                    row=dict(graph=graph,phase=ph,arm=arm,seed=b,Ra_seed=a,Rc_seed=c,stratum=s,direction=direction,query_count=len(ids),
                        unique_deleted_facts=len(set((ids//2).tolist())),unique_control_facts=len(set(cluster[ids//2].tolist())),**stats)
                    roles.append(intervals(row,v));reps.append(v)
            notify(arm,a,b)
    seed=aggregate(roles,reps,('graph','phase','arm','seed','stratum','direction'),2)
    mean=aggregate(roles,reps,('graph','phase','arm','stratum','direction'),6)
    # A paired bootstrap contrast, not subtraction of two separately computed CIs.
    contrast=[];cr=[];lookup={(r['arm'],r['Ra_seed'],r['seed'],r['stratum'],r['direction']):(r,v) for r,v in zip(roles,reps)}
    for a,b,c in itertools.permutations(cfg['seeds']):
        for s in cfg['strata']:
            for direction in cfg['directions']:
                left=lookup.get(('GraphDPO_B',a,b,s,direction));right=lookup.get(('M',a,b,s,direction))
                if left is None or right is None:continue
                l,lv=left;m,mv=right;point=None if l['AUC_minus_R'] is None or m['AUC_minus_R'] is None else l['AUC_minus_R']-m['AUC_minus_R']
                contrast.append(dict(graph=graph,phase=ph,seed=b,Ra_seed=a,stratum=s,direction=direction,query_count=l['query_count'],B_minus_M_excess_AUC=point))
                cr.append((lv[:,2]-mv[:,2])[:,None])
    cm=aggregate(contrast,cr,('graph','phase','stratum','direction'),6,('B_minus_M_excess_AUC',))
    noise=[]
    for s in cfg['strata']:
        for direction in cfg['directions']:
            group=[v for v in roles if v['arm']=='R' and v['stratum']==s and v['direction']==direction]
            for name in ('RR_primary_noise','Ra_selected_primary_noise_secondary','RR_aux_AUC','Ra_selected_aux_AUC_secondary'):
                arr=[v[name] for v in group if v[name] is not None];complete=len(arr)==6
                noise.append(dict(graph=graph,phase=ph,stratum=s,direction=direction,metric=name,roles_defined=len(arr),mean=float(np.mean(arr)) if complete else None,
                   mean_abs_from_0_or_half=float(np.mean(abs(np.asarray(arr)-(.5 if 'aux' in name else 0)))) if complete else None,
                   role_distribution_low=float(np.quantile(arr,.025)) if complete else None,role_distribution_high=float(np.quantile(arr,.975)) if complete else None,
                   note='role distribution is not CI; signed role exchange can cancel; Ra-selected secondary is selection-affected in bins'))
    return dict(auc_by_role=roles,auc_by_seed=seed,auc_mean=mean,noise_by_role=[r for r in roles if r['arm']=='R'],noise_summary=noise,B_minus_M_contrast_mean=cm)

def prediction(contrasts):
    result=[]
    for g in ('FB','WN'):
        v={r['stratum']:r for r in contrasts if (r['graph'],r['phase'],r['direction'])==(g,3,'combined')}
        names=('all','R_le10','R_11to100','R_gt100');known=all(s in v and v[s]['B_minus_M_excess_AUC'] is not None for s in names)
        point=bool(v['all']['B_minus_M_excess_AUC']>0 and v['R_le10']['B_minus_M_excess_AUC']>0 and v['R_le10']['B_minus_M_excess_AUC']>max(v[s]['B_minus_M_excess_AUC'] for s in names[2:])) if known else None
        result.append(dict(graph=g,point_prediction_matches=point,overall_B_minus_M_excess_AUC=v.get('all',{}).get('B_minus_M_excess_AUC'),
            high_recoverability_B_minus_M_excess_AUC=v.get('R_le10',{}).get('B_minus_M_excess_AUC'),
            note='Largest-bin operational check; AUC strata not additive; no privacy or significance certificate'))
    return result
