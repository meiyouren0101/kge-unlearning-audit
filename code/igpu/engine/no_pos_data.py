"""Unpaired deletion blocks; exact unchanged boundary/KD mask and RNG consumption."""
import random,types
import numpy as np
import torch
import io_utils as io
import legacy_runtime as rt
def mask_and_boundary_rng(args,kg):
    """Mirror only official RNG consumption; never form a positive triple."""
    io.require(args.random_policy=='far','Pinned main sampler policy far required')
    snap=kg.snapshots[args.snapshot];retained=list(snap.reserve_triples);random.shuffle(retained)
    adjacent={};mask=torch.zeros(kg.ent_num);designated=set(snap.forgotten_entities)
    for h,r,t in retained:
        adjacent.setdefault(h,[]).append((h,r,t));adjacent.setdefault(t,[]).append((h,r,t))
        if h in designated:mask[t]=1
        if t in designated:mask[h]=1
    draws=0
    for i,(h,r,t) in enumerate(snap.forgotten_triples):
        pool=adjacent.get(h if snap.is_heads[i] else t,[]);label=random.randint(0,kg.ent_num-1);draws+=1
        # The original far check returns after the first neighbour, including
        # its existing early-return behavior. Preserve it rather than fix it.
        count=0
        while pool and (pool[0][0]==label or pool[0][2]==label):
            label=random.randint(0,kg.ent_num-1);draws+=1;count+=1
            if count>10:break
    return mask,draws
class DeletedDataset(torch.utils.data.Dataset):
    def __init__(self,args,kg):
        self.args,self.kg=args,kg;self.neg_facts=[]
        for h,r,t in kg.snapshots[args.snapshot].forgotten_triples:self.neg_facts.extend(((h,r,t),(t,r+1,h)))
    def __len__(self):return len(self.neg_facts)
    def __getitem__(self,i):
        h,r,t=self.neg_facts[i];nh=np.random.randint(0,self.kg.ent_num-1,self.args.neg_ratio-1);nt=np.random.randint(0,self.kg.ent_num-1,self.args.neg_ratio-1);p=np.random.rand(self.args.neg_ratio-1)
        heads=np.where(p>.5,h,nh);tails=np.where(p>.5,nt,t)
        facts=[(h,r,t)]+[(int(a),r,int(b)) for a,b in zip(heads,tails)]
        return torch.LongTensor(facts),(h,r,t)
    @staticmethod
    def collate_fn(rows):
        facts=torch.cat([v[0] for v in rows]);neg=torch.LongTensor([v[1] for v in rows]);return facts[:,0],facts[:,1],facts[:,2],neg
def processor(up,args,kg):
    mask,draws=mask_and_boundary_rng(args,kg);ds=DeletedDataset(args,kg)
    loader,gen=rt._loader(ds,int(args.batch_size),int(args.seed))
    boundary=up.ForgetBoundaryDataset(args,kg);bl,bgen=rt._loader(boundary,int(args.batch_size),int(args.seed))
    return types.SimpleNamespace(dataset=ds,entity_distill_mask=mask,data_loader=loader,forget_boundary_dataset=boundary,forget_boundary_dataloader=bl,rng_burn_draws=draws)
