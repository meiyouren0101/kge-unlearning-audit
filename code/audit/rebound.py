"""Matched rebound leakage with fixed pre-probe strata and identical batch schedules."""
from itertools import permutations
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from igpu import bootstrap, audit
import common as c

def leak_stat(before, after, rb_before, rb_after, mask):
    count = int(mask.sum())
    if not count:
        return dict(query_count=0, method_change=None, R_change=None, Leak=None)
    a = float((1/after[mask]-1/before[mask]).mean())
    r = float((1/rb_after[mask]-1/rb_before[mask]).mean())
    return dict(query_count=count, method_change=a, R_change=r, Leak=a-r)

def compute(folders, out):
    registry = {}
    for folder in map(Path, folders):
        m = audit.verify(folder)
        registry[m['method'],m['seed']] = folder
    seeds = c.SEEDS
    import torch
    def schedule(folder):
        path = folder/'p3/probe/model.pt'
        expected = json.loads((folder/'COMPLETE.json').read_text('utf-8'))['files']['p3/probe/model.pt']
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Probe checkpoint hash mismatch')
        payload = torch.load(path,map_location='cpu',weights_only=True)['metadata']
        if not payload.get('new_empty_Adam'):
            raise ValueError('Probe Adam initialization not verified')
        return (payload['updates'],payload['schedule_sha256'])
    def ranks(folder):
        q,d,b,_ = audit.cache(folder,3,'primary')
        qs,ds,a,_ = audit.cache(folder,3,'probe')
        if not (np.array_equal(q,qs) and np.array_equal(d,ds)):
            raise ValueError('Rebound changed queries or cohorts')
        return q,d,b,a
    refs = {s:ranks(registry['R',s]) for s in seeds}
    ps = {s:audit.cache(registry['P',s],3,'primary') for s in seeds}
    baseline = {s:schedule(registry['R',s]) for s in seeds}
    rows, mean = [], []
    for method in sorted({m for m,s in registry} - {'P','R'}):
        for s in seeds:
            if schedule(registry[method,s]) != baseline[s]:
                raise ValueError('Probe schedule differs from the matched reference: '+method)
        xs = {s:ranks(registry[method,s]) for s in seeds}
        cells = {}
        for a,b,cseed in permutations(seeds):
            q,d,x0,x1 = xs[b]
            for v in (refs[a],refs[b],ps[b]):
                if not (np.array_equal(q,v[0]) and np.array_equal(d,v[1])):
                    raise ValueError('Rebound comparison query mismatch')
            ra = refs[a][2]
            _,_,rb0,rb1 = refs[b]
            for name, mask in [('all',d),('R_le10',d&(ra<=10)),('R_11to100',d&(ra>10)&(ra<=100)),
                               ('R_gt100',d&(ra>100)),('focus',d&(ra>100)&(ps[b][2]<=10))]:
                result = dict(method=method,R_bin=name,Ra=a,Rb=b,Rc=cseed,
                              **leak_stat(x0,x1,rb0,rb1,mask))
                rows.append(result); cells.setdefault(name, []).append(result['Leak'])
        for name, values in cells.items():
            mean.append(dict(method=method,R_bin=name,Leak=math.fsum(values)/6 if None not in values else None,roles=6))
    c.csvwrite(out/'leak_by_combination.csv',rows)
    c.csvwrite(out/'leak_mean.csv',mean)

