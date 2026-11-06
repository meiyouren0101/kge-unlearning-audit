import io_utils as io

def summaries(records,n_deleted,queries):
    import numpy as np
    x=np.asarray(records,dtype=np.float64);rows=[];directions=np.asarray([q[1]%2 for q in queries])
    io.require(len(x)==len(queries),'Rank/query count')
    for filt,col in (('legacy',0),('complete',1)):
        for cohort,offset,v in (('cumulative_forget',0,x[:n_deleted,col]),('all_retained',n_deleted,x[n_deleted:,col])):
            for direction in ('combined','forward','inverse'):
                a=v if direction=='combined' else v[directions[offset:offset+len(v)]==int(direction=='inverse')]
                rows.append(dict(filter=filt,cohort=cohort,direction=direction,count=len(a),MR=float(a.mean()) if len(a) else None,MRR=float((1/a).mean()) if len(a) else None,**{'Hit'+str(k):float((a<=k).mean()) if len(a) else None for k in (1,3,10)}))
    return rows

