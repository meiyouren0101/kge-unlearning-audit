"""Sequential ZeroFisher v2; no external job state or ranking-cache format required."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from igpu import bootstrap
import numpy as np
import torch
from igpu import audit, data as dat, run, training as tr
import common as c
import graph_data as gd
import legacy_runtime as rt
from . import author_adapter, zf_core as zf

ARMS = ('ZF', 'ZF-exact', 'ZF-official')
ROOT = Path(__file__).resolve().parents[1]

def select_scale(rows, p_mrr):
    eligible = [r for r in rows if r['status'] == 'COMPLETE' and r['retained_MRR'] >= .95*p_mrr]
    return (min(eligible, key=lambda r: (r['deleted_MRR'], -r['scale']))['scale'], False) if eligible else (10, True)

def finite(model):
    return all(bool(torch.isfinite(p).all()) for p in model.parameters())

def model_for(data, cfg, phase, seed, device, source):
    rt.seed_rng(seed)
    up, _ = tr.model_api(cfg)
    args = tr.args_for(cfg, data, phase, seed, device)
    model = up.TransE(args, gd.kg(data, phase))
    rt.load_model_state(model, source['model'])
    model.eval()
    return up, model

def load_p(folder, dataset, seed):
    folder = Path(folder)
    md = audit.verify(folder)
    if (md['dataset'], md['method'], md['seed'], md['evaluation']) != (dataset, 'P', seed, 'full'):
        raise ValueError('Scale and initial state require corresponding full P runs')
    path = folder / 'pretrained.pt'
    if hashlib.sha256(path.read_bytes()).hexdigest() != json.loads((folder/'COMPLETE.json').read_text('utf-8'))['files'][path.name]:
        raise ValueError('P checkpoint changed')
    return torch.load(path, map_location='cpu', weights_only=True)

def choose(data, cfg, device, p_folder, p, arm):
    retained = list(gd.queries(gd.at(data, 0)['retained']))
    deleted = list(gd.queries(data['phases'][0]))
    indexes = sorted(np.random.Generator(np.random.PCG64(2514013)).choice(len(retained), 10000, replace=False))
    sample = [retained[i] for i in indexes]
    keys, _, ranks, _ = audit.cache(p_folder, 0, 'primary')
    ranks_by_query = dict(zip(map(tuple, keys.tolist()), ranks))
    p_mrr = float(np.mean([1 / ranks_by_query[q] for q in sample]))
    up, model = model_for(data, cfg, 0, 0, device, p)
    rows, delta, info = zf.correction(up, model, data, 0, rt.schedule_seed(0,0), arm == 'ZF-exact')
    base = tuple(x.index_select(0, ids).detach() for x, ids in zip(model.embedding(), rows))
    candidates = []
    for scale in zf.SCALES:
        zf.apply(model, rows, delta, scale, base)
        norm = info['ihvp_l2']/scale
        row = dict(scale=scale, status='DIVERGED', retained_MRR=None, deleted_MRR=None,
                   update_l2=norm if math.isfinite(norm) else None)
        if finite(model):
            try:
                rr = run.raw_ranks(model, sample, data, cfg)[:, 4]
                dr = run.raw_ranks(model, deleted, data, cfg)[:, 4]
            except ValueError as e:
                if 'Nonfinite' not in str(e):
                    raise
            else:
                row.update(status='COMPLETE', retained_MRR=sum(1/int(r) for r in rr)/len(rr),
                           deleted_MRR=sum(1/int(r) for r in dr)/len(dr))
        candidates.append(row)
    eta, fallback = select_scale(candidates, p_mrr)
    return dict(scale=eta, fallback=fallback, P_retained_MRR=p_mrr, candidates=candidates,
                retained_sample_sha256=rt.tensor_tree_hash(sample))

def execute(data, device, p_folders, out):
    if data['dataset'] not in ('FB', 'WN'):
        raise ValueError('ZeroFisher audit roster is FB/WN TransE')
    if device.type != 'cuda':
        raise ValueError('The unmodified official arm requires CUDA')
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(4)
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((ROOT/'configs/GraphDPO.json').read_text('utf-8'))
    cfg['query_batch_size'] = 16
    p = {s:load_p(p_folders[s],data['dataset'],s) for s in c.SEEDS}
    selections = {a:choose(data,cfg,device,p_folders[0],p[0],a) for a in ARMS[:2]}
    selections[ARMS[2]] = dict(scale=10)
    c.write(out/'scale_selection.json', selections)
    c.csvwrite(out/'scale_grid.csv', [dict(method=a, **r) for a in ARMS[:2] for r in selections[a]['candidates']])
    binary = author_adapter.build(out)
    folders = []
    for arm in ARMS:
        for seed in c.SEEDS:
            folder = out/arm/str(seed); folder.mkdir(parents=True)
            c.write(folder/'run.json', dict(dataset=data['dataset'],method=arm,seed=seed,backbone='TransE',evaluation='full'))
            options = dict(cfg,name=arm)
            c.write(folder/'config.json', options)
            start = p[seed]
            for phase in range(4):
                stage = folder/f'p{phase}'; stage.mkdir()
                torch.cuda.reset_peak_memory_stats(device)
                up, model = model_for(data, options, phase, seed, device, start)
                rng = rt.schedule_seed(seed, phase); rt.seed_rng(rng)
                torch.cuda.synchronize(device)
                begin = time.perf_counter()
                scale = selections[arm]['scale']
                if arm == 'ZF-official':
                    first, second = author_adapter.make_loaders(binary, stage, data, phase, rng)
                    info = author_adapter.apply(model, first, second)
                    if info['score_calls'] != 8:
                        raise ValueError('Official first-batch score-call count')
                else:
                    rows, step, info = zf.correction(up, model, data, phase, rng, arm=='ZF-exact')
                    zf.apply(model, rows, step, scale)
                torch.cuda.synchronize(device)
                info.update(scale=scale, update_seconds=time.perf_counter()-begin,
                            gpu_peak_reserved_mib=torch.cuda.max_memory_reserved(device)/1048576)
                if not finite(model):
                    c.write(stage/'status.json',dict(status='DIVERGED', phase=phase));break
                c.write(stage/'training.json',info)
                start = dict(model=rt.cpu_tree(model.state_dict()), metadata=dict(kind='delete',seed=seed,phase=phase))
                rt.save_checkpoint_new(stage/'model.pt',start)
                pref = run.original_preferred(data,options,seed,phase,device)
                run.evaluate(start,data,options,seed,phase,device,stage,pref)
            else:
                c.write(folder/'COMPLETE.json', dict(files={f.relative_to(folder).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in folder.rglob('*') if f.is_file()}))
                folders.append(folder)
    return folders

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True)
    ap.add_argument('--p-runs',type=Path,nargs=3,required=True,metavar=('SEED0','SEED42','SEED123'))
    ap.add_argument('--r-runs',type=Path,nargs=3,required=True,metavar=('SEED0','SEED42','SEED123'))
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    data=dat.load(a.data)
    folders=execute(data,torch.device(a.device),dict(zip(c.SEEDS,a.p_runs)),a.out)
    audit.run([*a.p_runs,*a.r_runs,*folders],data,a.out/'audit')

if __name__=='__main__':
    main()

