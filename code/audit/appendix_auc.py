"""Rebuild Appendix C5 scores from the retained graph and label with existing R ranks."""
from pathlib import Path
import numpy as np
from igpu import bootstrap, audit
import graph_data as gd
import common as c
import features
from . import score_auc as auc

def compute(folders, data, out):
    registry = {}
    for folder in map(Path, folders):
        m = audit.verify(folder)
        if m['method'] != 'R':
            continue
        if m['dataset'] != data['dataset'] or m['seed'] in registry:
            raise ValueError('Unexpected or duplicate retraining reference')
        registry[m['seed']] = folder
    rows = []
    for phase in range(4):
        target, cohort, refs, sampled = None, None, {}, None
        for seed in auc.SEEDS:
            q, d, rank, meta = audit.cache(registry[seed],phase,'primary')
            if target is None:
                target,cohort,sampled = q,d,meta['cohort'] == 'fixed final retained sample'
            elif not (np.array_equal(target,q) and np.array_equal(cohort,d)):
                raise ValueError('R query or cohort mismatch')
            refs[seed] = rank[d]
        phase_data = gd.at(data,phase)
        _,_,s,_,_ = features.compute(data,phase)
        scores,_ = auc.prepare_scores(phase_data['retained'],phase_data['cumulative'],target[cohort],s)
        rows.extend(auc.evaluate(data['dataset'],phase,target[cohort],scores,refs,sampled=sampled))
    means = auc.aggregate(rows)
    c.csvwrite(out/'auc_by_seed.csv',rows); c.csvwrite(out/'auc_by_request.csv',means)
    summary = []
    for direction in ('forward','inverse'):
        for feature in auc.FEATURES:
            values = [r['AUC_mean'] for r in means if r['direction']==direction and r['feature']==feature]
            defined = len(values)==4 and None not in values
            summary.append(dict(dataset=data['dataset'],direction=direction,feature=feature,
                AUC_min_over_requests=min(values) if defined else None,
                AUC_max_over_requests=max(values) if defined else None,requests=len(values),sampled=sampled))
    c.csvwrite(out/'auc_range_over_requests.csv',summary)

