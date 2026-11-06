import io_utils as io

def matched_controls(facts,deleted,seed):
    import numpy as np
    df=facts[deleted];rf=facts[~deleted];pool={}
    for j,q in enumerate(rf):pool.setdefault(int(q[1]),[]).append(j)
    missing=sorted(set(map(int,df[:,1]))-set(pool));io.require(not missing,'MISSING_RELATION_CONTROLS: '+str(missing))
    rng=np.random.default_rng(seed);matched=np.asarray([pool[int(q[1])][int(rng.integers(len(pool[int(q[1])]))) ] for q in df],dtype=int)
    unique,cluster=np.unique(matched,return_inverse=True)
    io.require(np.array_equal(df[:,1],rf[matched,1]),'Relation matching identity')
    return matched,cluster,dict(deleted_facts=len(df),retained_pool_facts=len(rf),matched_draws=len(matched),unique_matched_retained_facts=len(unique),
        duplicate_control_draws=len(matched)-len(unique),matching_with_replacement=True,sampling_seed=seed)
