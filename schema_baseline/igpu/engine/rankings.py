"""Fixed-roster descriptive rank agreement; never treats six roles as replicates."""
from collections import defaultdict
import itertools
import math

KEYS=('graph','endpoint','cohort','direction','stratum')
def ranks(values):
    ordered=sorted(values,key=lambda k:(values[k],k));out={};i=0
    while i<len(ordered):
        j=i+1
        while j<len(ordered) and values[ordered[j]]==values[ordered[i]]:j+=1
        for arm in ordered[i:j]:out[arm]=(i+1+j)/2
        i=j
    return out

def agreement(a,b):
    names=sorted(set(a)&set(b));ra=ranks({k:a[k] for k in names});rb=ranks({k:b[k] for k in names})
    n=len(names);x=[ra[k] for k in names];y=[rb[k] for k in names]
    mx=sum(x)/n if n else 0;my=sum(y)/n if n else 0
    numerator=math.fsum((u-mx)*(v-my) for u,v in zip(x,y));den=math.sqrt(math.fsum((u-mx)**2 for u in x)*math.fsum((v-my)**2 for v in y))
    concordant=discordant=tied=0
    for u,v in itertools.combinations(names,2):
        z=(a[u]-a[v])*(b[u]-b[v])
        if z>0:concordant+=1
        elif z<0:discordant+=1
        else:tied+=1
    return dict(common_method_count=n,descriptive_spearman=numerator/den if den else None,
        concordant_method_pairs=concordant,discordant_method_pairs=discordant,tied_method_pairs=tied,
        no_pairwise_reversal=discordant==0 if n>=2 else None)

def make(seed_rows,mean_rows=()):
    available=[];groups=defaultdict(dict)
    for r in seed_rows:
        key=tuple(r[k] for k in KEYS);block=(r['phase'],r['seed'])
        values=groups[key].setdefault(block,{})
        if r['abs_log_mean'] is not None:values[r['arm']]=r['abs_log_mean']
    common_rows=[];stability=[];ranges=[]
    for key,blocks in groups.items():
        meta=dict(zip(KEYS,key))
        for (phase,seed),values in blocks.items():
            for arm,rank in ranks(values).items():available.append(dict(meta,phase=phase,seed=seed,arm=arm,rank=rank,abs_log_mean=values[arm],roster='available_in_this_block'))
        common=set.intersection(*(set(v) for v in blocks.values())) if blocks else set()
        fixed={block:{arm:v[arm] for arm in sorted(common)} for block,v in blocks.items()}
        for (phase,seed),values in fixed.items():
            for arm,rank in ranks(values).items():common_rows.append(dict(meta,phase=phase,seed=seed,arm=arm,rank=rank,abs_log_mean=values[arm],roster='intersection_all_admitted_phase_seed_blocks',common_methods='|'.join(sorted(common))))
        for left,right in itertools.combinations(sorted(fixed),2):
            if left[0]!=right[0] and left[1]!=right[1]:continue
            stability.append(dict(meta,comparison='across_seeds_same_phase' if left[0]==right[0] else 'across_phases_same_seed',
                phase_left=left[0],seed_left=left[1],phase_right=right[0],seed_right=right[1],common_methods='|'.join(sorted(common)),**agreement(fixed[left],fixed[right])))
        for arm in sorted(common):
            v=[ranks(v)[arm] for v in fixed.values()];ranges.append(dict(meta,arm=arm,blocks=len(v),rank_min=min(v),rank_max=max(v),rank_mean=sum(v)/len(v),top1_blocks=sum(x==1 for x in v),common_methods='|'.join(sorted(common))))
    phase_groups=defaultdict(dict);phase_rows=[]
    for row in mean_rows:
        key=tuple(row[k] for k in (*KEYS,'phase'))
        if row['abs_log_mean'] is not None:phase_groups[key][row['arm']]=row['abs_log_mean']
    for key,values in phase_groups.items():
        for arm,rank in ranks(values).items():phase_rows.append(dict(zip((*KEYS,'phase'),key),arm=arm,rank=rank,abs_log_mean=values[arm]))
    return dict(rankings_by_seed=available,rankings_by_phase=phase_rows,rankings_common_roster=common_rows,stability_pairwise=stability,rank_ranges=ranges)
