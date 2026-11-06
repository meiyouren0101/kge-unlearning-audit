"""Private Schema adapter, using the pinned original matcher."""
from pathlib import Path
import ast, copy, hashlib, importlib.util, random
from types import SimpleNamespace
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
UP=ROOT/'upstream'
SOURCE_SHA={
 'Exp_FB15K237/Entity_TransE_FB15K237.py':'30fbb9a9bb3cfcb100fb82d48a617c1c3444424754de1128296c1c903abdf445',
 'openke/data/EmbeddingDataLoader.py':'145a868a6369b49838a6c047dbf5737f5080e6bf2ac98bfdb04d0c42a2b557f8'}

def upstream_class():
    for name,h in SOURCE_SHA.items():
        if hashlib.sha256((UP/name).read_bytes()).hexdigest()!=h:raise ValueError('Upstream source changed: '+name)
    spec=importlib.util.spec_from_file_location('private_schema_loader',UP/'openke/data/EmbeddingDataLoader.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod.CosineSchemaDataLoader

def stage_lists(all_facts,requests,k):
    # Canonical input h,r,t -> OpenKE h,t,r, only current and earlier requests.
    if k not in range(4) or len(requests)!=4:raise ValueError('Exactly four requests required')
    truths=set(map(tuple,all_facts));deleted=[tuple(f) for q in requests[:k+1] for f in q]
    if len(truths)!=len(all_facts) or len(set(deleted))!=len(deleted) or not set(deleted)<=truths:raise ValueError('Invalid requests')
    gone=set(deleted)
    retained=[(h,t,r) for h,r,t in all_facts if (h,r,t) not in gone]
    current=[(h,t,r) for h,r,t in requests[k]]
    return retained,current

def loader_for(model,retained,current):
    cls=upstream_class();loader=cls.__new__(cls)
    loader.triples=np.asarray(retained,dtype=np.int64).reshape(-1,3)
    loader.removed_triples=np.asarray(current,dtype=np.int64).reshape(-1,3)
    loader.adj_matrix=loader.create_adj_matrix(loader.triples)
    loader.n_clusters=3
    loader.labels=loader.entity_cluster(model.ent_embeddings.weight.detach().cpu().numpy(),3)
    # No type_constrain file is needed: query_match_entity never uses self.schemas.
    return loader

def run_phase(model,all_facts,requests,k,seed,indexed=True,on_step=None,on_progress=None):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    retained,current=stage_lists(all_facts,requests,k);loader=loader_for(model,retained,current)
    model=model # intentionally inherit supplied endpoint, no reinitialization
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-5,weight_decay=0)
    by_relation={}
    for triple in loader.triples:by_relation.setdefault(int(triple[2]),[]).append(triple)
    updated=0;skipped=0;match_rows=[];losses=[]
    for offset,query in enumerate(loader.removed_triples):
        if on_progress is not None and offset%4096==0:on_progress(offset,len(current),updated)
        optimizer.zero_grad();e1=e2=e3=e4=None;attempts=0
        candidates=by_relation.get(int(query[2]),[]) if indexed else loader.triples
        while (e1 is None or e2 is None or e3 is None or e4 is None) and attempts<20:
            e1,e2,e3,e4=loader.query_match_entity(candidates,loader.adj_matrix,query,loader.labels);attempts+=1
        # Preserve upstream behavior even if the 20th call first succeeds.
        if attempts>=20:skipped+=1;match_rows.append(None);continue
        ids=[torch.tensor(int(v),dtype=torch.long,device=model.ent_embeddings.weight.device) for v in (e1,e2,e3,e4)]
        a=model.ent_embeddings(ids[0])-model.ent_embeddings(ids[1])
        b=model.ent_embeddings(ids[2])-model.ent_embeddings(ids[3])
        loss=1-torch.nn.functional.cosine_similarity(a,b,dim=0)
        if on_step is not None and not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite Schema loss')
        loss.backward();optimizer.step();updated+=1
        losses.append(loss.item());match_rows.append([int(x) for x in (e1,e2,e3,e4)])
        if on_step is not None and updated & (updated-1)==0:on_step(updated,model,optimizer)
    return dict(phase=k,requested=len(current),updates=updated,skipped=skipped,losses=losses,matches=match_rows,
                source='supplied previous endpoint',new_empty_Adam=True,actual_lr=1e-5,clusters=3,cluster_random_state=0)

class QuietProgress:
    def __init__(self,*args,**kw):pass
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def update(self,*args):pass
    def set_description(self,*args):pass

def reference_original_loop(model,loader,seed):
    """Execute the actual pinned driver's loop AST, excluding file/GPU setup."""
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    tree=ast.parse((UP/'Exp_FB15K237/Entity_TransE_FB15K237.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compute_embedding']
    nodes += [n for n in tree.body if isinstance(n,ast.With)]
    env=dict(torch=torch,device=model.ent_embeddings.weight.device,tqdm=QuietProgress,
             Cosine_model=model,Schema_dataloader=loader,total_loss=0.,
             trainer=SimpleNamespace(optimizer=torch.optim.Adam(model.parameters(),lr=1e-5,weight_decay=0)))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'pinned_official_schema_loop','exec'),env)
    return env['total_loss']
