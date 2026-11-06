"""Graph and sealed P/R rank inputs. External directories are read-only."""
from pathlib import Path
from hashlib import sha256
import zipfile
import numpy as np
from igpu import bootstrap
import graph_data as gd
import io_utils as io
import matrix20 as m

ROSTERS=('FB10','WN10','FB20','WN20')
TAGS={'FB':'fb15k-237','WN':'wn18rr'}
HISTORICAL=m.read(m.ROOT/'REFERENCES10.json')

def graph(key, path):
    tag=key[:2]
    if key.endswith('20'):
        m.verify_file(path,m.read(m.ROOT/'DATA_SHA256.json')[tag]['sha256'])
        return m.graph(tag,path)
    pins=m.read(m.ROOT/'DATA10_PINS.json')[tag]
    name=TAGS[tag]+'-10'
    members=['all_data.txt',*(f'{k}/{k}.txt' for k in range(4))]
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        m.check(len(set(names))==len(names),'Duplicate data archive entries')
        prefixes=[n[:-len('all_data.txt')] for n in names if n in (name+'/all_data.txt','all_data.txt')]
        m.check(len(prefixes)==1,'Expected official ZIP layout for '+name)
        raw={n:z.read(prefixes[0]+n) for n in members}
    for n in members:
        m.check(sha256(raw[n]).hexdigest()==pins['payload_sha256'][n],'Wrong pinned '+name+' input: '+n)
    data=gd.parse(raw)
    _,_,_,_,ne,nr,nt,_=m.SPECS[tag]
    m.check(data['counts']==dict(entities=ne,relations=nr,triples=nt,per_stage=pins['request_counts'][:4]),'Unexpected graph size')
    return dict(data,dataset=tag,name=name)

def require_available(path):
    if path.is_symlink():raise ValueError('Unexpected file symlink: '+str(path))
    if not path.is_file():raise m.Pending(str(path))
    return path

def names20(root,tag,seed,phase):
    base=Path(root)/tag/'stages'
    return [base/f'{prefix}_R_{seed}_p{phase}'/'commit.json' for prefix in ('train','evaluate')]

def waiting(key,phase,a20):
    """Check only expected seals. No inspection of other jobs, pointers, or logs."""
    if key.endswith('10'):
        tag=key[:2]
        paths=[Path(r['commit_path']) for arm in ('P','R') for s in m.SEEDS
               for r in HISTORICAL[tag][f'{arm}/{s}/{phase}']['receipts']]
    else:
        paths=[p for s in m.SEEDS for p in names20(a20,key[:2],s,phase)]
    return [str(p) for p in dict.fromkeys(paths) if not p.is_file()]

def sealed_result(seal, known_seal=None, known_result=None):
    require_available(seal)
    if known_seal:m.verify_file(seal,known_seal)
    obj=m.read(seal);rpath=io.child(seal.parent,'result.json')
    m.check(obj['status'] in ('COMPLETE','COMMITTED'),'Stage seal is not terminal: '+str(seal))
    h=obj['files']['result.json']
    if known_result:m.check(h==known_result,'Pinned result not bound by its seal')
    result=m.read(m.verify_file(rpath,h))
    m.check(result['status']=='COMPLETE','Reference stage did not complete: '+str(seal))
    return obj,result

def checkpoint_digest(result):
    ck=result.get('checkpoint')
    return ck['sha256'] if isinstance(ck,dict) else result.get('checkpoint_sha256')

def historical_source(tag,arm,seed,phase):
    row=HISTORICAL[tag][f'{arm}/{seed}/{phase}']
    old=Path(row['cache']['path']);h=row['cache']['sha256'];found=[]
    for pin in row['receipts']:
        seal=Path(pin['commit_path'])
        m.check(seal.parent/'result.json'==Path(pin['result_path']),'Mismatched historical receipt paths')
        commit,result=sealed_result(seal,pin['commit_sha256'],pin['result_sha256'])
        if commit['files'].get(old.name)==h:
            m.check((result['arm'],result['seed'],result.get('phase',phase))==(arm,seed,phase),'Historical rank cache identity')
            if arm=='P':
                m.check(checkpoint_digest(result)==m.read(m.ROOT/'P_PINS.json')[tag][str(seed)]['sha256'],'P ranking and checkpoint mismatch')
            found.append((seal,result))
    m.check(len(found)==1,'Historical cache needs exactly one pinned evaluation seal')
    seal,result=found[0];relocated=seal.with_name(old.name)
    path=relocated if relocated.is_file() else old
    if isinstance(result.get('cache'),dict):
        m.check(result['cache']['sha256']==h,'Historical cache result hash mismatch')
    # Read only the exact historical cache or its identical resumed copy.
    m.verify_file(path,h)
    return path,dict(cache_sha256=h,seal=str(seal),commit_sha256=m.digest(seal),producer='historical',path=str(path))

def a_source(root,tag,seed,phase):
    train,ev=names20(root,tag,seed,phase)
    m.check(not train.is_symlink() and not ev.is_symlink(),'Unexpected A seal symlink')
    s,r=sealed_result(train);t,v=sealed_result(ev)
    for value,path,mode in ((r,train,'reference'),(v,ev,'evaluate')):
        m.check((value['tag'],value['arm'],value['seed'],value['phase'],value['mode'],value['name'])==
                (tag,'R',seed,phase,mode,path.parent.name),'Wrong A v2 reference identity')
    model_sha=s['files']['final.pt']
    m.check(r['checkpoint_sha256']==model_sha==v['source_checkpoint_sha256'],'A evaluation not bound to its R training')
    path=io.child(ev.parent,'ranks.u32le');h=t['files']['ranks.u32le'];m.verify_file(path,h)
    info=m.read(m.verify_file(io.child(ev.parent,'evaluation.json'),t['files']['evaluation.json']))
    m.check((info['method'],info['seed'],info['phase'],info['query_sha256'])==
            ('R',seed,phase,v['query_sha256']),'Wrong A R evaluation metadata')
    return path,dict(cache_sha256=h,seal=str(ev),commit_sha256=m.digest(ev),producer='A_v2',
                     path=str(path),model_sha256=model_sha,query_sha256=info['query_sha256'],
                     n_deleted=info['n_deleted'])

def canonical(data,phase):
    d=gd.at(data,phase)
    deleted=list(gd.queries(d['deleted_order']))
    return np.asarray(deleted+list(gd.queries(d['retained'])),dtype='<u4'),len(deleted)

def align(array,query,n_deleted,entities):
    m.check(array.shape==(len(query),5) and len(query)>n_deleted>0,'Wrong cache row count')
    m.check(bool(((array[:,3:]>0)&(array[:,3:]<=entities)).all()),'Rank out of bounds')
    if np.array_equal(array[:,:3],query):return array
    parts=[]
    for a,q in ((array[:n_deleted],query[:n_deleted]),(array[n_deleted:],query[n_deleted:])):
        order=lambda x:np.lexsort((x[:,2],x[:,1],x[:,0]))
        src=order(a);dest=order(q)
        m.check(np.array_equal(a[src,:3],q[dest]),'Query identity or retained/deleted cohort mismatch')
        b=np.empty_like(a);b[dest]=a[src];parts.append(b)
    return np.concatenate(parts)

def stage_plan(key):
    tag=key[:2];out=[]
    def add(name,mode,arm,seed,k,needs=(),**kw):
        out.append(dict(name=name,mode=mode,arm=arm,seed=seed,phase=k,tag=tag,needs=list(needs),
                        estimated_peak_mib=2048,**kw))
    for k in range(4):
        for seed in m.SEEDS:
            n=f'Schema_{seed}_p{k}'
            previous=f'train_Schema_{seed}_p{k-1}' if k else None
            if key.endswith('20'):add(f'evaluate_P_{seed}_p{k}','evaluate','P',seed,k)
            add(f'train_{n}','delete','Schema',seed,k,[previous] if previous else (),previous=previous)
            add(f'evaluate_{n}','evaluate','Schema',seed,k,[f'train_{n}'],source=f'train_{n}')
    return out
