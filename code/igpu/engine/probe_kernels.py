"""Verbatim matched alternative probe math; no server dependency intake."""
from collections import Counter,defaultdict
import hashlib,math,random
import graph_data as gd
import io_utils as io

def seed_value(seed,phase):
    return int.from_bytes(hashlib.sha256(f'GraphDPO-fair-probe/20261026/{seed}/{phase}'.encode()).digest()[:8],'little')

def quotas(counts,total):
    n=sum(counts.values());wanted=min(total,n);q={k:counts[k]*wanted//n for k in counts}
    for k in sorted(counts,key=lambda k:(-(counts[k]*wanted%n),k))[:wanted-sum(q.values())]:q[k]+=1
    return q

def build_one(data,phase,seed,original,forbidden,limit):
    import numpy as np
    queries=[(t,r+1,h) if head else (h,r,t) for (h,r,t),head in zip(original['facts'],original['is_heads'])]
    strata=defaultdict(list)
    for i,q in enumerate(queries):strata[q[1]].append(i)
    count={k:len(v) for k,v in strata.items()};quota=quotas(count,limit);rng=random.Random(seed_value(seed,phase))
    ids=sorted(i for k in sorted(strata) for i in rng.sample(strata[k],quota[k]));independent=[];actual=[];used=set();rows=[]
    truths=gd.filters(gd.at(data,phase)['retained'])['complete'];unique_truths={};owners=[]
    for j,index in enumerate(ids):
        h,r,t=queries[index];chosen=None
        for _ in range(10000):
            candidate=(h,r,rng.randrange(data['counts']['entities']))
            if candidate not in forbidden and candidate not in used:chosen=candidate;break
        if chosen is None:
            allowed=[(h,r,a) for a in range(data['counts']['entities']) if (h,r,a) not in forbidden and (h,r,a) not in used]
            io.require(bool(allowed),'No independent unobserved answer left in matched context; do not relax exclusion')
            chosen=rng.choice(allowed)
        independent.append(chosen);actual.append(queries[index]);used.add(chosen)
        other_answers=[answer for answer in sorted(truths.get((h,r),())) if answer!=t]
        for answer in other_answers:
            q=(h,r,answer)
            if q not in unique_truths:unique_truths[q]=len(unique_truths)
            owners.append((j,unique_truths[q]))
        rows.append(dict(anchor_id=j,request_index=index,head=h,relation=r,direction='inverse' if r%2 else 'forward',
                         actual_preferred_answer=t,independent_answer=chosen[2],retained_true_answer_count=len(other_answers)))
    return dict(independent=np.asarray(independent,dtype='<u4'),actual=np.asarray(actual,dtype='<u4'),
       truth=np.asarray(list(unique_truths),dtype='<u4').reshape(-1,3),truth_owners=np.asarray(owners,dtype='<u4').reshape(-1,2),rows=rows,
       strata=[dict(directed_relation=k,direction='inverse' if k%2 else 'forward',population=count[k],sampled=quota[k]) for k in sorted(count)])

def metrics(values,queries,owners=None):
    import numpy as np
    v=np.asarray(values,dtype=float);q=np.asarray(queries)
    rows=[]
    for direction in ('combined','forward','inverse'):
        ids=np.arange(len(v)) if direction=='combined' else np.flatnonzero(q[:,1]%2==int(direction=='inverse')) if len(q) else np.array([],dtype=int)
        a=v[ids];rows.append(dict(direction=direction,count=len(a),MRR=float((1/a).mean()) if len(a) else None,Hit10=float((a<=10).mean()) if len(a) else None))
    return rows

def truth_request_metrics(ranks,plan):
    import numpy as np
    groups=defaultdict(list)
    for anchor,target in plan['truth_owners'].tolist():groups[anchor].append(ranks[target])
    rows=[]
    for direction in ('combined','forward','inverse'):
        wanted=[v for k,v in groups.items() if direction=='combined' or int(plan['independent'][k,1])%2==int(direction=='inverse')]
        rows.append(dict(direction=direction,count=len(wanted),MRR=float(np.mean([np.mean(1/np.asarray(v,float)) for v in wanted])) if wanted else None,
                         Hit10=float(np.mean([np.mean(np.asarray(v)<=10) for v in wanted])) if wanted else None))
    return rows

