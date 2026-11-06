"""Negative-only margin + Eq.5; no positive argument or positive scoring path."""
import math
import torch
import torch.nn.functional as F
class _Gate(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w):ctx.save_for_backward(w);return x.view_as(x)
    @staticmethod
    def backward(ctx,g):return g*ctx.saved_tensors[0],None
def gate(x,w):
    if x.shape!=w.shape or w.requires_grad or not bool(torch.isfinite(w).all()) or not bool(((w>=0)&(w<=2)).all()):raise ValueError('Finite detached G2 weights in [0,2] required')
    return _Gate.apply(x,w)
def from_vectors(grid,current_negative,old_negative,w,margin=8.):
    if grid.ndim!=2 or grid.shape[1]!=10 or grid.shape[0]!=w.numel():raise ValueError('Ten-negative blocks required')
    if current_negative.shape!=w.shape or old_negative.shape!=w.shape or old_negative.requires_grad:raise ValueError('Negative/reference shapes and detached reference required')
    deleted=grid[:,0];negatives=torch.cat((gate(deleted,w).unsqueeze(1),grid[:,1:]),1)
    ml=F.relu(margin-negatives.mean(1)).sum()/(11*len(grid))
    logneg=F.logsigmoid(-current_negative)
    dp=-.1*F.logsigmoid(-(gate(logneg,w)-F.logsigmoid(-old_negative))).mean()
    return ml,dp,(deleted,logneg)
def losses(model,batch,w):
    h,r,t,neg=batch;n=neg.shape[0]
    if h.numel()!=10*n or w.shape!=(n,) or not torch.equal(torch.stack((h,r,t),1).reshape(n,10,3)[:,0],neg):raise ValueError('Deleted slot0 layout')
    ent,rel=model.embedding();old_ent,old_rel=model.old_embeddings()
    grid=model.score_fun(ent.index_select(0,h),rel.index_select(0,r),ent.index_select(0,t)).reshape(n,10)
    def dist(e,z):return model.score_fun(e.index_select(0,neg[:,0]),z.index_select(0,neg[:,1]),e.index_select(0,neg[:,2]))
    ml,dp,channels=from_vectors(grid,dist(ent,rel),dist(old_ent,old_rel),w,float(model.args.margin))
    kd=model.distill_loss();total=ml.float();total+=dp;total+=kd
    return total,dict(margin=ml.float(),dpo=dp,distill=kd),channels
def norm(values):return math.sqrt(math.fsum(float(v.detach().square().sum().cpu()) for v in values))
def step(model,opt,batch,weights,rt,device):
    model.train();batch=rt._batch_to_device(batch,device);params=tuple(model.parameters());opt.zero_grad()
    w=torch.tensor(weights,dtype=torch.float64,device=device);total,parts,ch=losses(model,batch,w)
    coeff=torch.autograd.grad(total,ch,retain_graph=True)
    grads=torch.autograd.grad(ch,params,grad_outputs=tuple(v.detach() for v in coeff),retain_graph=True)
    if not bool(torch.isfinite(total)) or not all(bool(torch.isfinite(g).all()) for g in grads):raise ValueError('Nonfinite deletion gradient')
    diag=dict(deleted_gradient_l2=norm(grads),deleted_entity_gradient_l2=norm(grads[:1]),deleted_relation_gradient_l2=norm(grads[1:]),gradient_status='MEASURED_EXPLICIT_DELETE_CHANNELS')
    numeric=dict(total=float(total.detach().cpu()),**{k:float(v.detach().cpu()) for k,v in parts.items()})
    total.backward()
    if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params):raise ValueError('Nonfinite parameter gradient')
    diag['total_gradient_l2']=norm([p.grad for p in params]);opt.step();return numeric,diag
