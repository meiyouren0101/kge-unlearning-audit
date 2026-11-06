"""Frozen evaluation-only cohorts, shared by all arms/seeds and all four phases."""
from collections import defaultdict
from pathlib import Path
import hashlib
import random
import struct
import common as c
import graph_data as gd
import io_utils as io
Q=struct.Struct('<III');RECORD=struct.Struct('<IIIII')
DEFAULT=dict(seed=20261024,retained_query_limit=50000,deleted_fact_limit_per_phase=10000)


def policy():return {k:io.read(c.ROOT/'protocol.json')['evaluation']['sampling'][k] for k in DEFAULT}


def rng(seed,tag,stream):
    digest=hashlib.sha256(f'{seed}|{tag}|{stream}'.encode('utf-8')).digest()
    return random.Random(int.from_bytes(digest,'big'))


def digest_queries(queries):
    h=hashlib.sha256()
    for q in queries:h.update(Q.pack(*q))
    return h.hexdigest()


def stratified(queries,limit,generator):
    """Proportional directed-relation strata, largest remainder, integer arithmetic."""
    groups=defaultdict(list)
    for q in sorted(queries):groups[q[1]].append(q)
    n=sum(map(len,groups.values()));k=min(limit,n)
    if not n:return [],[]
    quotas={r:k*len(v)//n for r,v in groups.items()}
    remainders=sorted(groups,key=lambda r:(-(k*len(groups[r])%n),r))
    for r in remainders[:k-sum(quotas.values())]:quotas[r]+=1
    result=[];strata=[]
    for r,values in sorted(groups.items()):
        chosen=values if quotas[r]==len(values) else generator.sample(values,quotas[r])
        result.extend(chosen)
        strata.append(dict(directed_relation=r,population=len(values),sample=quotas[r],
                           inclusion_probability=quotas[r]/len(values)))
    return sorted(result),strata


def build(data,tag,spec=None):
    spec=dict(DEFAULT if spec is None else spec)
    final=gd.at(data,3)['retained'];population=list(gd.queries(final))
    retained,strata=stratified(population,spec['retained_query_limit'],rng(spec['seed'],tag,'retained'))
    deleted=[]
    for phase,facts in enumerate(data['phases']):
        candidates=sorted(range(len(facts)),key=lambda i:facts[i])
        k=min(spec['deleted_fact_limit_per_phase'],len(candidates))
        chosen=candidates if k==len(candidates) else rng(spec['seed'],tag,f'deleted_p{phase}').sample(candidates,k)
        chosen=sorted(chosen,key=lambda i:facts[i])
        deleted.append(dict(phase=phase,population_facts=len(facts),sample_facts=k,
                            request_indices=chosen,query_sha256=digest_queries(gd.queries([facts[i] for i in chosen]))))
    return dict(schema_version=2,dataset=tag,policy=spec,source_counts=data['counts'],
        cohort_label='fixed final-retained cohort (sampled)',
        retained_population_queries=len(population),retained_sample_queries=len(retained),
        retained_queries=retained,retained_strata=strata,retained_query_sha256=digest_queries(retained),
        deleted_by_phase=deleted,
        isolation='evaluation only; not full current-retained population; no sampling input to training/gating/selection')


def prepare(session,data,tag,spec=None):
    session=Path(session);manifest=build(data,tag,policy() if spec is None else spec);files={}
    def save(name,rows):
        path=session/name;c.csvwrite(path,rows);files[name]=io.sha(path)
    save('evaluation_retained_queries.csv',[
        dict(query_id=i,h=h,r=r,t=t,direction='inverse' if r%2 else 'forward')
        for i,(h,r,t) in enumerate(manifest['retained_queries'])])
    for row in manifest['deleted_by_phase']:
        phase=row['phase'];rows=[]
        for i in row['request_indices']:
            for h,r,t in gd.queries([data['phases'][phase][i]]):
                rows.append(dict(request_index=i,deletion_phase=phase,h=h,r=r,t=t,
                                 direction='inverse' if r%2 else 'forward'))
        save(f'evaluation_deleted_p{phase}_queries.csv',rows)
    save('evaluation_retained_strata.csv',manifest['retained_strata'])
    manifest['query_lists_sha256']=files;c.write(session/'evaluation_samples.json',manifest)
    return dict(manifest_sha256=io.sha(session/'evaluation_samples.json'),query_lists_sha256=files)


def load(session,data):
    session=Path(session);manifest=io.read(session/'evaluation_samples.json')
    io.require(manifest['source_counts']==data['counts'],'Evaluation population counts changed')
    io.require(manifest['policy']==policy(),'Evaluation sampling policy changed')
    pin=session/'stages/admit/evaluation_sample_pin.json'
    if pin.is_file():
        expected=io.read(pin);io.require(io.sha(session/'evaluation_samples.json')==expected['manifest_sha256'],'Evaluation sample manifest changed')
        io.require(manifest['query_lists_sha256']==expected['query_lists_sha256'],'Evaluation query pins changed')
        if (pin.parent/'commit.json').exists():c.stage(session,'admit')
    for name,h in manifest['query_lists_sha256'].items():io.require(io.sha(session/name)==h,'Evaluation query list changed '+name)
    manifest['retained_queries']=[tuple(q) for q in manifest['retained_queries']]
    return manifest


def select(manifest,data,phase):
    deleted=[];sources=[]
    for row in manifest['deleted_by_phase'][:phase+1]:
        facts=[data['phases'][row['phase']][i] for i in row['request_indices']]
        block=list(gd.queries(facts));deleted.extend(block);sources.extend([row['phase']]*len(block))
    queries=deleted+list(manifest['retained_queries'])
    return dict(queries=queries,n_deleted=len(deleted),source_phases=sources+[-1]*len(manifest['retained_queries']),
                directions=[q[1]%2 for q in queries],query_sha256=digest_queries(queries),
                sampling=dict(cohort_label=manifest['cohort_label'],seed=manifest['policy']['seed'],
                    retained_population_queries=manifest['retained_population_queries'],
                    retained_sample_queries=manifest['retained_sample_queries'],
                    cumulative_deleted_sample_queries=len(deleted),
                    cumulative_deleted_population_queries=2*sum(len(x) for x in data['phases'][:phase+1]),
                    current_retained_population_queries=2*len(gd.at(data,phase)['retained']),
                    manifest_sha256=None))


def details(session,data,phase):
    result=select(load(session,data),data,phase)
    result['sampling']['manifest_sha256']=io.sha(Path(session)/'evaluation_samples.json')
    return result


def get(session,data,phase):return details(session,data,phase)['queries']


def preferred_indices(session,data,phase):return load(session,data)['deleted_by_phase'][phase]['request_indices']
