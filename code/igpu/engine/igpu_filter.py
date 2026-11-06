"""Universal M-only target availability rule; no alternative loss or placeholder."""
from collections import Counter
import math
import graph_data as gd
import io_utils as io

def omissions(retained,requests):
    counts=Counter(q[1] for q in retained)
    return [i for i,q in enumerate(requests) if not counts[q[1]]]

def check_cap(omitted,total):
    io.require(total>0 and omitted*1000<=total,'IGPU empty-positive omissions exceed0.1% of a deletion request; stop')

def apply(data,phase,proc):
    """Change ONLY main dataset lists, after original sampler/replay/mask creation."""
    d=gd.at(data,phase);requests=data['phases'][phase];heads=data['is_heads'][phase]
    missing=omissions(d['retained'],requests);check_cap(len(missing),len(requests))
    before=len(proc.dataset);drop=set(missing);kept=[i for i in range(len(requests)) if i not in drop]
    if missing:
        rows=[j for i in kept for j in (2*i,2*i+1)]
        for field in ('facts','pos_facts','neg_facts'):
            values=getattr(proc.dataset,field);io.require(len(values)==2*len(requests),'Original M augmented row identity')
            setattr(proc.dataset,field,[values[j] for j in rows])
    # Existing RandomSampler queries len(dataset) dynamically, no replacement/reseed.
    io.require(len(proc.dataset)==2*len(kept),'No placeholder rows')
    io.require(len(proc.data_loader)==math.ceil(len(proc.dataset)/1024),'Main loader follows actual retained row count')
    receipt=dict(rule='Only M: empty same-relation same-side retained-positive pool => omit original fact from deletion batches; keep evaluation unchanged',
        original_request_count=len(requests),omitted_fact_count=len(missing),omitted_forward_request_indices=missing,
        omitted_augmented_rows=2*len(missing),actual_main_rows=len(proc.dataset),original_main_rows=before,
        actual_batches_per_epoch=len(proc.data_loader),normalization='unchanged original losses, actual batch rows; no placeholder/zero loss rows',
        omitted_facts=[dict(request_index=i,fact=list(requests[i]),designated_answer_direction='inverse' if heads[i] else 'forward',
            omitted_training_directions=['forward','inverse']) for i in missing],
        evaluation_unchanged=True,KD_mask_and_boundary_replay_unchanged=True,new_loss=False,threshold_fraction=.001)
    current=[requests[i] for i in kept];side=[heads[i] for i in kept]
    return dict(d,phases=[[],[],current],is_heads=[[],[],side]),receipt
