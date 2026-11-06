"""Exactly unchanged forward values, fixed gradient multipliers including G2>1."""
import math
import torch
import torch.nn.functional as F
import objective

class _Gate(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w):ctx.save_for_backward(w);return x.view_as(x)
    @staticmethod
    def backward(ctx,g):return g*ctx.saved_tensors[0],None

def gate(x,w):
    if x.shape!=w.shape or w.requires_grad or not bool(torch.isfinite(w).all()) or not bool(((w>=0)&(w<=2)).all()):
        raise ValueError('Fixed finite weights in [0,2] required')
    return _Gate.apply(x,w)

def from_vectors(grid,cp,cn,op,on,w,margin=8.):
    if grid.ndim!=2 or grid.shape[1]!=11 or grid.shape[0]!=w.numel():raise ValueError('11-row blocks')
    deleted=grid[:,1];negatives=torch.cat((gate(deleted,w).unsqueeze(1),grid[:,2:]),1)
    y=torch.tensor([-1.],device=grid.device,dtype=torch.float32)
    ml=F.margin_ranking_loss(grid[:,0],negatives.mean(1),y,margin=margin,reduction='sum')/grid.numel()
    logneg=F.logsigmoid(-cn)
    diff=F.logsigmoid(-cp)-F.logsigmoid(-op)-gate(logneg,w)+F.logsigmoid(-on)
    dp=-.1*F.logsigmoid(diff).mean()
    return ml,dp,(deleted,logneg)

def losses(model,batch,w):
    h,r,t,label,pos,neg=batch;n=pos.shape[0]
    if h.numel()!=11*n or w.shape!=(n,):raise ValueError('Main batch layout')
    labels=label.reshape(n,11)
    if not bool((labels[:,0]==1).all() and (labels[:,1:]==-1).all()):raise ValueError('Main label layout')
    if not torch.equal(torch.stack((h,r,t),1).reshape(n,11,3)[:,1],neg):raise ValueError('Explicit deletion slot1')
    ent,rel=model.embedding()
    grid=model.score_fun(ent.index_select(0,h),rel.index_select(0,r),ent.index_select(0,t)).reshape(n,11)
    ml,dp,channels=from_vectors(grid,*objective.distance_vectors(model,pos,neg),w,float(model.args.margin))
    kd=model.distill_loss();total=ml.float();total+=dp;total+=kd
    return total,dict(margin=ml.float(),dpo=dp,distill=kd),channels

def norm(values):return math.sqrt(math.fsum(float(v.detach().square().sum().cpu()) for v in values))

def step(model,opt,batch,weights,rt,device,unit_check=False):
    model.train();batch=rt._batch_to_device(batch,device);params=tuple(model.parameters());opt.zero_grad()
    w=torch.tensor(weights,dtype=torch.float64,device=device)
    if unit_check:
        unit,_,_=losses(model,batch,torch.ones_like(w));h,r,t,l,pos,neg=batch
        old=model.loss(h,r,t,l).float();old+=model.dpo_loss(pos,neg);old+=model.distill_loss()
        if not torch.equal(old,unit):raise ValueError('Unit forward bridge')
        g1=torch.autograd.grad(unit,params);g0=torch.autograd.grad(old,params)
        if not all(torch.equal(x,y) for x,y in zip(g1,g0)):raise ValueError('Unit parameter-gradient bridge')
    total,components,channels=losses(model,batch,w)
    coeff=torch.autograd.grad(total,channels,retain_graph=True)
    gd=torch.autograd.grad(channels,params,grad_outputs=tuple(g.detach() for g in coeff),retain_graph=True)
    if not bool(torch.isfinite(total)) or not all(bool(torch.isfinite(g).all()) for g in gd):raise ValueError('Nonfinite deletion gradient')
    diag=dict(deleted_gradient_l2=norm(gd),deleted_entity_gradient_l2=norm(gd[:1]),deleted_relation_gradient_l2=norm(gd[1:]),
        gradient_status='MEASURED_EXPLICIT_DELETE_CHANNELS')
    numeric=dict(total=float(total.detach().cpu()),**{k:float(v.detach().cpu()) for k,v in components.items()})
    total.backward()
    if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params):raise ValueError('Nonfinite parameter gradient')
    diag['total_gradient_l2']=norm([p.grad for p in params]);opt.step()
    return numeric,diag
