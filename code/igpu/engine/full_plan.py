"""Fill only old T fallback positions with retained truths; independent local RNG."""
from collections import defaultdict,Counter
import hashlib
import random
from target_plan import augmented,digest,ARMS
import io_utils as io


def stream_seed(seed,head):
    tag='head' if head else 'tail'
    return int.from_bytes(hashlib.sha256(f'graphdpo-T-full-v1/20261022/{seed}/fallback/{tag}'.encode()).digest()[:8],'little')


def build(data,old_plan,seed):
    retained=set(map(tuple,data['retained']));deleted=set(map(tuple,data['cumulative']))
    io.require(not retained&deleted,'Retained/deleted overlap')
    pools=defaultdict(list);fresh=defaultdict(list)
    for fact in sorted(retained):
        io.require(fact[1]%2==0,'Forward retained pool')
        pools[fact[1]].append(fact)
        if fact not in old_plan.replay:fresh[fact[1]].append(fact)
    rng={head:random.Random(stream_seed(seed,head)) for head in (False,True)}
    targets=list(old_plan.positives[ARMS[2]]);eligible=set(old_plan.eligible);counts=Counter();rows=[]
    for i,request in enumerate(old_plan.requests):
        old=targets[i];head=old_plan.heads[i]
        if i in eligible:rule='keep_old_T_replacement'
        else:
            relation=request[1]
            candidates=fresh[relation]
            rule='fallback_same_relation_unreplayed'
            if not candidates:candidates=pools[relation];rule='fallback_same_relation_all_retained'
            io.require(bool(candidates),'No retained facts of requested relation; stop, no cross-relation repair')
            targets[i]=rng[head].choice(candidates)
        io.require(targets[i] in retained and targets[i] not in deleted and targets[i][1]==request[1],
                   'T_full targets must be same-relation retained forward truths')
        if i in eligible:io.require(targets[i]==old,'Old eligible T target changed')
        counts[rule]+=1
        rows.append({'request_index':i,'designated_head':head,'rule':rule,'target_changed_from_T':targets[i]!=old,
                     'head':targets[i][0],'relation':targets[i][1],'tail':targets[i][2]})
    receipt={'seed':seed,'forward_requests':len(targets),'old_target_plan':old_plan.receipt,
        'rule_counts':{k:counts[k] for k in ('keep_old_T_replacement','fallback_same_relation_unreplayed','fallback_same_relation_all_retained')},
        'fallback_positions':len(targets)-len(eligible),'changed_target_count':sum(r['target_changed_from_T'] for r in rows),
        'all_targets_retained':True,'all_relations_preserved':True,'eligible_targets_unchanged':True,
        'target_array_sha256':digest(targets),'augmented_target_array_sha256':digest(augmented(targets)),
        'independent_stream_seeds':{('head' if h else 'tail'):stream_seed(seed,h) for h in (False,True)}}
    return targets,rows,receipt


def apply(dataset,targets,requests):
    io.require(dataset.neg_facts==augmented(requests),'Original rejected requests differ')
    dataset.pos_facts=augmented(targets)
    dataset.facts=[{'fact':x,'label':1} for x in dataset.pos_facts]
    return digest(dataset.pos_facts)
