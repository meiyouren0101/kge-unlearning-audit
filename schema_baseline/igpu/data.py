"""Pinned official splits and deletion requests; local inputs or explicit download."""
from . import bootstrap
from pathlib import Path
import argparse,hashlib,io,json,tarfile,urllib.request,zipfile
import graph_data as gd
ROOT=Path(__file__).resolve().parents[1]
PINS=Path(__file__).resolve().parent/'assets/SOURCE_PINS.json'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def fetch(url):
    with urllib.request.urlopen(url,timeout=120) as r:return r.read()
def prepare(dataset,out,archive=None,splits=None,download=False):
    pins=json.loads(PINS.read_text('utf-8'));spec=pins['datasets'][dataset]
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    if archive is None:
        if not download:raise ValueError('Provide --requests-zip or explicitly allow --download')
        raw=fetch(pins['requests'][dataset]['url'])
    else:raw=Path(archive).read_bytes()
    if sha(raw)!=pins['requests'][dataset]['sha256']:raise ValueError('Pinned request ZIP SHA256 mismatch')
    req={}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for k in range(5):
            names=[n for n in z.namelist() if n.endswith('/'+f'{k}/{k}.txt')]
            if len(names)!=1:raise ValueError('Missing/ambiguous request member')
            req[k]=z.read(names[0])
    blob=None;tar=None
    if splits is None and dataset in ('FB','WN'):
        if not download:raise ValueError('Supply --official-splits containing train/valid/test.txt or allow --download')
        blob=fetch(spec['source_url'])
        if dataset=='WN' and sha(blob)!=spec['archive_sha256']:raise ValueError('Official WN archive hash')
    pieces=[];sources={}
    for name in ('train','valid','test'):
        pin=spec['files'][name]
        if splits is not None:raw=(Path(splits)/(name+'.txt')).read_bytes()
        elif dataset in ('CL','YA'):
            if not download:raise ValueError('Supply official CoDEx-L splits or --download')
            raw=fetch(pin['url'])
        elif dataset=='FB':
            with zipfile.ZipFile(io.BytesIO(blob)) as z:raw=z.read(pin['archive_member'])
        else:
            with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as z:
                matches=[n for n in z.getnames() if n.lstrip('./')==pin['archive_member']]
                if len(matches)!=1:raise ValueError('Ambiguous official split')
                raw=z.extractfile(matches[0]).read()
        if sha(raw)!=pin['sha256']:raise ValueError('Official split SHA256 mismatch: '+name)
        lines=[]
        for line in raw.decode('utf-8').splitlines():
            q=line.split('\t')
            if len(q)!=3 or not all(q):raise ValueError('Three nonempty TAB-separated strings required')
            lines.append('\t'.join(q)+'\n')
        pieces.append(''.join(lines).encode('utf-8'));sources[name]=sha(raw)
    all_data=b''.join(pieces)
    if sha(all_data)!=pins['all_data_sha256'][dataset]:raise ValueError('Rebuilt all_data SHA256 mismatch')
    truth={tuple(x.split('\t')) for x in all_data.decode('utf-8').splitlines()};seen=set();counts=[]
    if len(truth)!=len(all_data.splitlines()):raise ValueError('Official combined truth duplicates')
    for k,raw in req.items():
        facts=[]
        for line in raw.decode('utf-8').splitlines():
            q=line.split('\t')
            if len(q)!=4 or q[3] not in (q[0],q[2]):raise ValueError('Request designated-side format')
            facts.append(tuple(q[:3]))
        if len(set(facts))!=len(facts) or not set(facts)<=truth or seen&set(facts):raise ValueError('Request duplicate/subset/overlap')
        seen.update(facts);counts.append(len(facts))
        p=out/str(k);p.mkdir();(p/(str(k)+'.txt')).write_bytes(raw)
    (out/'all_data.txt').write_bytes(all_data)
    receipt=dict(dataset=dataset,rule='train->valid->test; original strings; TAB/LF; no sorting or deduplication',all_data_sha256=sha(all_data),
        official_split_sha256=sources,request_counts=counts,request4='admission only; four training requests0..3',
        payload_sha256={f.relative_to(out).as_posix():sha(f.read_bytes()) for f in out.rglob('*.txt')})
    expected=json.loads(PINS.with_name('PREPARED_SHA256.json').read_text('utf-8'))[dataset]
    if receipt['payload_sha256']!=expected:raise ValueError('Prepared payload hashes differ from release inventory')
    (out/'data_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');return receipt
def load(folder):
    folder=Path(folder);m=json.loads((folder/'data_manifest.json').read_text('utf-8'))
    for n,h in m['payload_sha256'].items():
        if sha((folder/n).read_bytes())!=h:raise ValueError('Prepared data changed: '+n)
    raw={n:(folder/n).read_bytes() for n in ('all_data.txt','0/0.txt','1/1.txt','2/2.txt','3/3.txt')}
    d=gd.parse(raw);d['dataset']=m['dataset'];d['name']={'FB':'fb15k-237-10','WN':'wn18rr-10','CL':'CoDEx-L-10','YA':'YAGO3-10-10','TOY':'fb15k-237-10'}[m['dataset']]
    return d
def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=('FB','WN','CL','YA'),required=True);p.add_argument('--out',required=True)
    p.add_argument('--requests-zip');p.add_argument('--official-splits');p.add_argument('--download',action='store_true');a=p.parse_args()
    print(json.dumps(prepare(a.dataset,a.out,a.requests_zip,a.official_splits,a.download),indent=2))
if __name__=='__main__':main()
