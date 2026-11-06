"""Retained-graph-only features and AUC plan. No file IO, ML or deletion inputs."""
from collections import Counter,defaultdict
import math
import io_utils as io

FEATURES=('f1_twohop_undirected','f2_other_direct','f3_relation_outdegree','f3_relation_indegree','f4_max_rule_confidence')

class RetainedFeatures:
    def __init__(self,retained):
        facts=list(map(tuple,retained));io.require(len(facts)==len(set(facts)),'Duplicate retained fact')
        self.neighbors=defaultdict(set);self.pairs=defaultdict(set);self.out=Counter();self.incoming=Counter()
        for h,r,t in facts:
            io.require(all(type(v) is int and v>=0 for v in (h,r,t)) and r%2==0,'Forward retained facts only')
            if h!=t:self.neighbors[h].add(t);self.neighbors[t].add(h)
            self.pairs[h,t].add(r);self.pairs[t,h].add(r+1)
        self.denominator=Counter();self.same=Counter();self.reverse=Counter()
        for (h,t),bodies in self.pairs.items():
            for r in bodies:self.out[h,r]+=1;self.incoming[r,t]+=1;self.denominator[r]+=1
            reversed_targets=self.pairs.get((t,h),())
            for body in bodies:
                for target in bodies:
                    if body!=target:self.same[body,target]+=1
                for target in reversed_targets:
                    if (body^1)!=target:self.reverse[body,target]+=1
        self.fact_count=len(facts)

    def row(self,q):
        h,r,t=q
        common=self.neighbors.get(h,set())&self.neighbors.get(t,set())
        paths=0 if h==t else len(common-{h,t})
        same=self.pairs.get((h,t),());opposite=self.pairs.get((t,h),())
        direct=any(x//2!=r//2 for x in same) or any(x//2!=r//2 for x in opposite)
        confidence=0.
        for body in same:
            if body!=r:confidence=max(confidence,self.same[body,r]/self.denominator[body])
        for body in opposite:
            if (body^1)!=r:confidence=max(confidence,self.reverse[body,r]/self.denominator[body])
        return (paths,int(direct),self.out[h,r],self.incoming[r,t],confidence)

def roc_auc(labels,scores):
    """Mann-Whitney AUC with exact half-credit for ties; no direction flipping."""
    io.require(len(labels)==len(scores)>0 and set(labels)<={0,1},'AUC inputs')
    io.require(all(math.isfinite(float(x)) for x in scores),'Nonfinite AUC score')
    pos=sum(labels);neg=len(labels)-pos
    if not pos or not neg:return None
    pairs=sorted(zip(scores,labels));sum_positive_ranks=0.;i=0
    while i<len(pairs):
        j=i+1
        while j<len(pairs) and pairs[j][0]==pairs[i][0]:j+=1
        rank=((i+1)+j)/2
        sum_positive_ranks+=rank*sum(y for _,y in pairs[i:j]);i=j
    return (sum_positive_ranks-pos*(pos+1)/2)/(pos*neg)

def masks(phases,ra,p,subset):
    io.require(len(phases)==len(ra)==len(p) and subset in (False,True),'Label mask lengths/scope')
    train=[];test=[];excluded={'train_middle':0,'test_middle':0,'train_P_excluded':0,'test_P_excluded':0}
    for i,(phase,r,pr) in enumerate(zip(phases,ra,p)):
        io.require(phase in (0,1,2) and r>=1 and pr>=1,'Label bounds')
        name='test' if phase==2 else 'train'
        if subset and pr>10:excluded[name+'_P_excluded']+=1;continue
        if 10<r<=100:excluded[name+'_middle']+=1;continue
        (test if phase==2 else train).append(i)
    return train,test,excluded

def transform_row(row):
    io.require(len(row)==5 and all(math.isfinite(x) and x>=0 for x in row),'Feature bounds')
    io.require(row[1] in (0,1) and row[4]<=1,'Binary feature/confidence bounds')
    return (math.log1p(row[0]),row[1],math.log1p(row[2]),math.log1p(row[3]),row[4])
