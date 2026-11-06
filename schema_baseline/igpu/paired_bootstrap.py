"""Fact-paired contrasts of absolute rank gaps; shared weights across all six roles."""
from . import bootstrap
from pathlib import Path
import argparse,itertools,json,math
import numpy as np
import common as c,bootstrap_core as bc
from . import audit,data as dat

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--data',required=True);p.add_argument('--left',default='IGPU');p.add_argument('--right',default='GraphDPO+repair');p.add_argument('--phase',type=int,default=3);p.add_argument('--replicates',type=int,default=2000);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);data=dat.load(a.data);reg={}
    for f in map(Path,a.runs):md=audit.verify(f);reg[md['method'],md['seed']]=f
    phase=a.phase;values={};keys=None;dm=None
    for m in (a.left,a.right,'P','R'):
        for seed in (0,42,123):
            q,d,r,e=audit.cache(reg[m,seed],phase,'primary')
            if keys is None:keys=q;dm=d
            elif not np.array_equal(q,keys) or not np.array_equal(d,dm):raise ValueError('Bootstrap query identity mismatch')
            values[m,seed]=r
    fids={q:i for i,q in enumerate(data['triples'])};ids=np.array([fids[(int(t),int(r)-1,int(h)) if r%2 else (int(h),int(r),int(t))] for h,r,t in keys])
    columns=bc.Columns(out,len(fids));groups=[]
    for cohort in ('cumulative_forget','all_retained'):
        for stratum in ('all','R_le10','R_11to100','R_gt100','focus_Pb_le10_Ra_gt100'):
            components=[];point=[]
            for x,b,control in itertools.permutations((0,42,123)):
                ra=values['R',x];rb=values['R',b];pb=values['P',b];mask=dm.copy() if cohort=='cumulative_forget' else ~dm
                if stratum=='R_le10':mask&=ra<=10
                elif stratum=='R_11to100':mask&=(ra>10)&(ra<=100)
                elif stratum=='R_gt100':mask&=ra>100
                elif stratum.startswith('focus'):mask&=(ra>100)&(pb<=10)
                if not mask.any():point=[];components=[];break
                an=np.abs(np.log(values[a.left,b][mask])-np.log(rb[mask]));bn=np.abs(np.log(values[a.right,b][mask])-np.log(rb[mask]));weights=np.ones(mask.sum())
                key=(cohort,stratum,x,b,control)
                components.append(tuple(columns.add((key,label),ids[mask],v) for label,v in (('an',an),('ad',weights),('bn',bn),('bd',weights))))
                point.append(float(an.mean()-bn.mean()))
            groups.append(dict(cohort=cohort,R_bin=stratum,point=math.fsum(point)/6 if len(point)==6 else None,components=components))
    matrix=columns.close();reps,digest=bc.resample(matrix,groups,a.replicates,20261028)
    rows=[]
    for j,g in enumerate(groups):
        low,high,n=bc.percentile(reps[:,j]);rows.append(dict(left=a.left,right=a.right,phase=phase,cohort=g['cohort'],R_bin=g['R_bin'],point=g['point'],CI95_low=low,CI95_high=high,defined=n))
    c.csvwrite(out/'paired_bootstrap.csv',rows);(out/'bootstrap.json').write_text(json.dumps(dict(replicates=a.replicates,seed=20261028,weights_sha256=digest,
        definition='Original forward fact is resampling unit; both directions/all seeds/all six roles/strata share weights; within-role difference then six-role mean. Conditional on this graph/seeds.'),indent=2)+'\n')
    matrix._mmap.close()
if __name__=='__main__':main()
