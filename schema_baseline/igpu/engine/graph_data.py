"""Original graph math; removed server asset-receipt loader."""
import types
import io_utils as io

def parse(raw):
    entities={};relations={};triples=[]
    def eid(x):
        if x not in entities:entities[x]=len(entities)
        return entities[x]
    for line in raw['all_data.txt'].decode('utf-8').splitlines():
        if not line.strip():continue
        h,r,t=line.split()
        if r not in relations:relations[r]=2*len(relations)
        triples.append((eid(h),relations[r],eid(t)))
    io.require(len(triples)==len(set(triples)),'Duplicate graph facts');truth=set(triples);seen=set();phases=[];heads=[];designated=[]
    for phase in range(4):
        facts=[];hs=[];es=[]
        for line in raw[f'{phase}/{phase}.txt'].decode('utf-8').splitlines():
            if not line.strip():continue
            h,r,t,e=line.split('\t');io.require(e in (h,t),'Designated side')
            q=(entities[h],relations[r],entities[t]);io.require(q in truth and q not in seen,'Request identity/overlap')
            facts.append(q);hs.append(e!=t);es.append(entities[e]);seen.add(q)
        phases.append(facts);heads.append(hs);designated.append(es)
    return dict(triples=triples,phases=phases,is_heads=heads,designated=designated,
        counts=dict(entities=len(entities),relations=len(relations),triples=len(triples),per_stage=list(map(len,phases))))

def at(data,phase):
    deleted=set(q for items in data['phases'][:phase+1] for q in items)
    return dict(data,cumulative=deleted,deleted_order=[q for q in data['triples'] if q in deleted],retained=[q for q in data['triples'] if q not in deleted])

def kg(data,phase):
    snapshots={}
    for j in range(phase+1):
        d=at(data,j);snapshots[j]=types.SimpleNamespace(forgotten_triples=data['phases'][j],forgotten_entities=data['designated'][j],
            is_heads=data['is_heads'][j],reserve_triples=d['retained'],accumulate_forgotten_triples=list(d['cumulative']),
            forgotten_num=len(data['phases'][j]),class_to_forget=data['designated'][j])
    return types.SimpleNamespace(ent_num=data['counts']['entities'],rel_num=2*data['counts']['relations'],train_data=data['triples'],snapshots=snapshots)

def queries(facts):
    for h,r,t in facts:yield h,r,t;yield t,r+1,h

def filters(triples):
    full={};old={}
    for h,r,t in triples:
        full.setdefault((h,r),set()).add(t);full.setdefault((t,r+1),set()).add(h)
        old.setdefault((h,r),set()).add(t);old[t,r+1]={h}
    return dict(complete=full,legacy=old)
