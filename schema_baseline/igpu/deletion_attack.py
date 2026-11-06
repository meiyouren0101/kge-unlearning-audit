"""CPU-only matched deleted-versus-retained attack with the experimental kernel."""
from . import bootstrap
from pathlib import Path
import argparse,json,types
import numpy as np
import common as c,attack_kernels as ak,attack_matching as matching
from . import audit

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--out',required=True);p.add_argument('--replicates',type=int,default=1000);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);reg={};graph=None
    aliases={'IGPU':'M','GraphDPO':'GraphDPO_B','GraphDPO-code':'GraphDPO_A','GraphDPO+repair':'B_repair'}
    for f in map(Path,a.runs):
        md=audit.verify(f)
        if graph is not None and graph!=md['dataset']:raise ValueError('One graph per audit')
        graph=md['dataset'];reg[aliases.get(md['method'],md['method']),md['seed']]=f
    if graph not in ('FB','WN','TOY'):raise ValueError('This attack requires the full paired main-evaluation cohort')
    for method in ('P','R'):
        if any((method,s) not in reg for s in (0,42,123)):raise ValueError('Missing three independent P/R reference seeds')
    arms=sorted({m for m,s in reg if m not in ('P','R','M')});tables={}
    cfg=dict(seeds=[0,42,123],strata=['all','R_le10','R_11to100','R_gt100'],directions=['combined','forward','inverse'],
        bootstrap_replicates=a.replicates,bootstrap_seed=2026102703,baseline_arms=arms,method_arms=['M'] if ('M',0) in reg else [])
    for phase in range(4):
        values={};facts=None;deleted=None
        for key,f in reg.items():
            raw=np.fromfile(f/f'p{phase}/ranks.u32le',dtype='<u4').reshape(-1,2,5);meta=json.loads((f/f'p{phase}/evaluation.json').read_text('utf-8'))
            if not np.array_equal(raw[:,0,0],raw[:,1,2]) or not np.array_equal(raw[:,0,2],raw[:,1,0]) or not np.array_equal(raw[:,0,1]+1,raw[:,1,1]):raise ValueError('Full reciprocal fact pairs required')
            q=raw[:,0,:3];order=np.lexsort((q[:,2],q[:,1],q[:,0]));d=order<meta['n_deleted']//2
            if facts is None:facts=q[order];deleted=d
            elif not np.array_equal(facts,q[order]) or not np.array_equal(deleted,d):raise ValueError('Query/cohort mismatch')
            values[key]=raw[order,:,4].astype(float)
        matched,cluster,receipt=matching.matched_controls(facts,deleted,ak.stable_seed(2026102702,graph,phase))
        dest=out/f'p{phase}';dest.mkdir();loader=types.SimpleNamespace(deleted=deleted)
        result=ak.phase(graph,phase,values,set(values),loader,matched,cluster,receipt,cfg,dest,lambda *x:None)
        for name,rows in result.items():tables.setdefault(name,[]).extend(rows)
        (dest/'matching.json').write_text(json.dumps(receipt,indent=2)+'\n')
    for name,rows in tables.items():c.csvwrite(out/(name+'.csv'),rows)
    (out/'definitions.json').write_text(json.dumps(dict(primary='AUC_X - AUC_Rb, signed; larger means more separable',
        noise='R_c minus R_b, independent of bucket selector R_a',auxiliary='X versus R_b ranks on same deleted facts; ideal0.5',
        bootstrap='paired fact-cluster multinomial, deleted weights and reused retained-control cluster weights, reciprocal directions shared',
        replicates=a.replicates,warning='Fixed-graph conditional uncertainty, not independent dataset or certified privacy guarantees'),indent=2)+'\n')
if __name__=='__main__':main()
