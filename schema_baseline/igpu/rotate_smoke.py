"""CPU-only seven-configuration RotatE exercise on a tiny fabricated graph."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from . import bootstrap
from pathlib import Path
import argparse,json,time
import torch
from . import smoke,run,audit
ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    start=time.perf_counter();out=Path(args.out);out.mkdir(parents=True,exist_ok=False);data=smoke.make_data(out/'data');runs=[]
    methods=('P','R','GraphDPO','IGPU','FT-lr1e-4','IGPU-half-repair','IGPU-no-repair')
    for name in methods:
        for seed in (0,42,123):
            cfg=json.loads((ROOT/'configs/RotatE'/(name+'.json')).read_text('utf-8'))
            cfg.update(embedding_dim=8,pretrain_epochs=2,delete_epochs=2)
            dest=out/name/str(seed);run.run(cfg,data,seed,torch.device('cpu'),dest,smoke=True);runs.append(dest)
    audit.run(runs,data,out/'audit')
    report=dict(status='PASS',CPU_only=True,GPU=False,backbone='RotatE',methods=methods,seeds=[0,42,123],requests=4,
        seconds=time.perf_counter()-start,note='Artificial D8/P2/R2/delete2/diagnostic2; formal configs unchanged; sampled evaluation kept separate.')
    (out/'SMOKE_RESULT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()
