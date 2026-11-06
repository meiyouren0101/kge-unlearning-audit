"""Value-equivalent streaming of the pinned official Fisher accumulator.

Never merge loss gradients, change original data batches, fix the original scaler,
or change its b_id1/3 accumulation. Only discard unused higher-order graphs.
"""
from pathlib import Path
import hashlib

ORIGINAL_PROCESS_SHA='6b3142f1b4831f3ab545cc5b039dbd7295d4c5a012dd8ebb15af354d224033d6'
NOTE='Streaming Fisher preserves the operation order. On three-seed tiny-graph checks, Fisher values and final model parameters had maximum error 0. Chunking only organizes the candidate-entity loop; the original per-fact batch remains 1, and gradients are not merged before squaring.'

def settings(session):
    import json
    p=session/'fisher_stream_settings.json'
    return json.loads(p.read_text('utf-8'))['class_chunk'] if p.exists() else 64

def descriptor(chunk=64):
    return dict(implementation='fisher_stream_v4',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                original_process_sha256=ORIGINAL_PROCESS_SHA,class_chunk=chunk,original_fact_batch=1,
                accumulation_batches=[1,3],model_update_rule_unchanged=True,note=NOTE,
                local_validation='CPU torch2.6.0; CUDA equivalence not measured locally')

def install(up, chunk=64):
    if not isinstance(chunk,int) or chunk<=0:raise ValueError('Positive class chunk required')
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    ns=up.FisherTrainBatchProcessor.hessian.__globals__
    autocast=ns['autocast'];tqdm=ns['tqdm']
    def hessian(self,dataset,model):
        model.eval()
        train_loader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
        loss_fn=nn.CrossEntropyLoss()
        for p in model.parameters():p.grad_acc=0;p.grad2_acc=0
        MAX_TRAIN_STEP=3;accumulation_steps=2
        for b_id,batch in tqdm(enumerate(self.data_loader)):
            if b_id>MAX_TRAIN_STEP:break
            bh,br,bt,by=batch
            bh=bh.to(self.args.device);br=br.to(self.args.device)
            bt=bt.to(self.args.device);by=by.to(self.args.device)
            with autocast():
                with torch.no_grad():
                    output=model.predict(bh,br)
                    prob=F.softmax(output,dim=-1)
                    output_new=torch.argmax(output,dim=1)
                for start in range(0,output.shape[1],chunk):
                    for y in range(start,min(start+chunk,output.shape[1])):
                        target=torch.empty_like(bt).fill_(y)
                        loss=model.loss(bh,br,target,by if by is not None else by).float()
                        model.zero_grad()
                        self.scaler.scale(loss).backward(retain_graph=True)
                        if (b_id+1)%accumulation_steps==0:
                            with torch.no_grad():
                                for p in model.parameters():
                                    if p.requires_grad:
                                        p.grad_acc+=(bt==target).sum().float()*p.grad.data
                                        p.grad2_acc+=prob[:,y].sum().float()*p.grad.data*p.grad.data
                            model.zero_grad()
        for p in model.parameters():
            p.grad_acc/=len(train_loader);p.grad2_acc/=len(train_loader)
    up.FisherTrainBatchProcessor.hessian=hessian
    return up

def hook_official_api(base, chunk):
    import official_api
    path=base/'official/src/model/model_process.py'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=ORIGINAL_PROCESS_SHA:
        raise ValueError('Official Fisher source identity changed')
    original=official_api.active
    def active(*args,**kwargs):
        up,sources=original(*args,**kwargs)
        return install(up,chunk),sources
    official_api.active=active
