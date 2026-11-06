"""Local and sealed inputs for the controlled GraphDPO 20% matrix."""
from pathlib import Path
from hashlib import sha256, sha1
from io import BytesIO
from itertools import chain
from collections import Counter
import json, zipfile

ROOT=Path(__file__).resolve().parent
SEEDS=(0,42,123)
COMMIT='2b0aa9597fc0c9a4b122f21d2935265ad60da1f7'
SPECS={
 'FB':('fb15k-237-20',7831790,'9542af7970a26bb9ad97692f69bf5e7d518c218e','5a355d584b6a6726a05d9f36479d88bfaa4fbcbaf4354869aae69f332889f0db',14541,237,310116,62023),
 'WN':('wn18rr-20',1841420,'11e7062e1c474f81e1b532b64cdcd438a2ed66b7','8cedafb8f7042e1e54ad8a6fb0a07964d02a2c823952c864874a97d75853ccf8',40943,11,93003,18600)}
METHODS=('Finetune','FT-lr1e-4','RL','NG','Fisher','BS','ADV-IMP','SSD','GraphDPO','GraphDPO-code','ZF','IGPU')
CONFIG={'ZF':'P','Fisher':'Fisher-stream'}
SCALES=(.0001,.0003,.001,.003,.01,.03,.1,.3,1,3,10,30,100)
class Pending(Exception): pass

def check(x,msg):
    if not x:raise ValueError(msg)
def digest(path):
    h=sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):
    def no_duplicate(pairs):
        value=dict(pairs)
        check(len(value)==len(pairs),'Duplicate JSON keys')
        return value
    return json.loads(Path(p).read_text('utf-8-sig'),object_pairs_hook=no_duplicate)
def verify_file(p,h):
    p=Path(p)
    check(p.is_file() and not p.is_symlink() and digest(p)==h,'Unverified input: '+str(p))
    return p

def signed_payload(folder):
    return {f.relative_to(folder).as_posix():digest(f) for f in sorted(folder.rglob('*')) if f.is_file() and f.name!='commit.json'}

def pin_p(tag,seed):
    row=read(ROOT/'P_PINS.json')[tag][str(seed)];p=Path(row['path']);seal=p.with_name('commit.json')
    if not seal.is_file() or not p.is_file():raise Pending(str(seal))
    r=row.get('receipt')
    if r:
        verify_file(seal,r['commit_sha256']);verify_file(r['result_path'],r['result_sha256'])
    m=read(seal)
    check(m['status'] in ('COMPLETE','COMMITTED') and m['files'][p.name]==row['sha256'],'Incorrect original P seal')
    return verify_file(p,row['sha256']),row['sha256']

def transfer(tag,base):
    root=Path(base).absolute()/tag;f=root/'stages/scale_ZF';seal=f/'commit.json'
    if not seal.is_file():raise Pending(str(seal))
    m=read(seal);check(m['status'] in ('COMPLETE','COMMITTED'),'Unsealed eta producer')
    run=read(root/'segment.json');check(run['dataset']==tag,'Scale dataset mismatch')
    top=root.parent
    if not (top/'complete.json').is_file():raise Pending(str(top/'complete.json'))
    complete=read(top/'complete.json'); admission=read(top/'admit.json')
    check(complete['status']=='COMPLETE' and admission['planned_stages']==154,'Wrong or unsealed 10% ZF producer')
    verify_file(root/'stages/summary/commit.json',complete['summary_seals'][tag])
    verify_file(top/'SCALE_RULE.txt',complete['scale_rule_sha256'])
    verify_file(ROOT/'SOURCE_10PCT_SCALE_RULE.txt',complete['scale_rule_sha256'])
    result=verify_file(f/'result.json',m['files']['result.json']);r=read(result)
    check(r['status']=='COMPLETE' and r['scale'] in SCALES,'Incorrect frozen eta receipt')
    check([x['scale'] for x in r['grid']]==list(SCALES),'Wrong eta grid')
    p,h=pin_p(tag,0)
    check(r['model_input']['sha256']==h,'Frozen eta used another P')
    return dict(scale=r['scale'],policy='transfer',fallback=r['fallback'],old_result=str(result),
                old_result_sha256=digest(result),old_commit_sha256=digest(seal))

def graph(tag,path):
    from igpu import bootstrap
    import graph_data as gd
    from igpu.engine import igpu_filter
    name,n,blob,all_sha,ne,nr,nt,nf=SPECS[tag]
    raw=Path(path).read_bytes()
    check(len(raw)==n and sha1(f'blob {n}\0'.encode()+raw).hexdigest()==blob,'Not the pinned official '+name+' ZIP')
    with zipfile.ZipFile(BytesIO(raw)) as z:
        req=[f'{k}/{k}.txt' for k in range(5)]
        check(len(z.namelist())==len(set(z.namelist())),'Duplicate ZIP members')
        payload={p:z.read(name+'/'+p) for p in ['all_data.txt',*req]}
    check(sha256(payload['all_data.txt']).hexdigest()==all_sha,'20% all-data identity differs from the 10% P graph')
    check([len(payload[p].splitlines()) for p in req]==[nf]*5,'Changed official request counts')
    # Fifth file is verified as part of the published blob, never parsed into graph or training.
    d=gd.parse({k:v for k,v in payload.items() if k!='4/4.txt'})
    check(d['counts']==dict(entities=ne,relations=nr,triples=nt,per_stage=[nf]*4),'20% graph dimensions')
    return dict(d,dataset=tag,name=name)

def matrix(tag,eta_policy):
    rows=[]
    def add(n,m,a,s,k,needs=(),**kw):
        rows.append(dict(name=n,mode=m,arm=a,seed=s,phase=k,needs=list(needs),
                         estimated_peak_mib=2560 if a=='ZF' else 2048,**kw))
    for k in range(4):
        for s in SEEDS:
            add(f'train_R_{s}_p{k}','reference','R',s,k)
            add(f'evaluate_R_{s}_p{k}','evaluate','R',s,k,[f'train_R_{s}_p{k}'],source=f'train_R_{s}_p{k}')
            add(f'evaluate_P_{s}_p{k}','evaluate','P',s,k)
    for arm in METHODS:
        if arm=='ZF':
            add('scale_ZF','grid' if eta_policy=='grid' else 'transfer','ZF',0,0,['evaluate_P_0_p0'] if eta_policy=='grid' else [])
        for s in SEEDS:
            for k in range(4):
                end=f'{arm}_{s}_p{k}'
                last=f'{arm}_{s}_p{k-1}'
                final=f'repair_{last}' if arm=='IGPU' else f'train_{last}'
                prev=[final]+([f'train_{last}'] if arm=='IGPU' else []) if k else []
                deps=prev+(['scale_ZF'] if arm=='ZF' else [])
                add(f'train_{end}','delete',arm,s,k,deps,previous=final if k else None,previous_delete=f'train_{last}' if k else None)
                source=f'train_{end}'
                if arm=='IGPU':
                    add(f'repair_{end}','repair',arm,s,k,[source],source=source);source=f'repair_{end}'
                add(f'evaluate_{end}','evaluate',arm,s,k,[source],source=source)
    return rows

def select_scale(rows,p):
    ok=[r for r in rows if r['status']=='COMPLETE' and r['retained_MRR']>=.95*p]
    return (min(ok,key=lambda r:(r['deleted_MRR'],-r['scale']))['scale'],False) if ok else (10,True)
