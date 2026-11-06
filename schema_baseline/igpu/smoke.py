"""Artificial CPU-only four-request exercise, including six-role audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from . import bootstrap
from pathlib import Path
import argparse,hashlib,json,time
import torch
from . import run,data as dat,audit
ROOT=Path(__file__).resolve().parents[1]
def make_data(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    facts=[(i,'r0',(i+1)%12) for i in range(12)]+[(i,'r1',(i+3)%12) for i in range(12)]
    encode=lambda i:'e'+str(i)
    all_data=''.join(f'{encode(h)}\t{r}\t{encode(t)}\n' for h,r,t in facts)
    (folder/'all_data.txt').write_text(all_data,encoding='utf-8')
    for phase in range(4):
        f=folder/str(phase);f.mkdir();a,b=facts[phase],facts[phase+12]
        (f/(str(phase)+'.txt')).write_text(f'{encode(a[0])}\t{a[1]}\t{encode(a[2])}\t{encode(a[0])}\n'+f'{encode(b[0])}\t{b[1]}\t{encode(b[2])}\t{encode(b[2])}\n',encoding='utf-8')
    m=dict(dataset='TOY',payload_sha256={f.relative_to(folder).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in folder.rglob('*.txt')})
    (folder/'data_manifest.json').write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8');return dat.load(folder)
def small_config(method):
    cfg=json.loads((ROOT/'configs'/(method+'.json')).read_text('utf-8'))
    cfg.update(embedding_dim=8,pretrain_epochs=2,delete_epochs=2)
    return cfg
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--quick',action='store_true');a=p.parse_args()
    start=time.perf_counter();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);data=make_data(out/'data');runs=[]
    methods=('IGPU',) if a.quick else ('P','R','IGPU')
    for method in methods:
        for seed in ((0,) if a.quick else (0,42,123)):
            dest=out/method/str(seed);run.run(small_config(method),data,seed,torch.device('cpu'),dest,smoke=True);runs.append(dest)
    audit.run(runs,data,out/'audit')
    result=dict(status='PASS',device='cpu',GPU=False,artificial=True,methods=list(methods),seconds=time.perf_counter()-start,
        note='Emb8,pretrain2,delete2,diagnostic2; formal configs remain emb200/100/20 and fixed diagnostic budgets.')
    (out/'SMOKE_RESULT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
