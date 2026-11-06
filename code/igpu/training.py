"""One method, one seed, four sequential requests; no scheduling or admission."""
from . import bootstrap
import copy,hashlib,json,math,os
from pathlib import Path
import torch
import common as c,graph_data as gd,legacy_runtime as rt,official_api
import training_kernels as tk,features,gate_core,gate_sensitivity,rank_kernels as rk

def model_api(cfg,include_attack=False):
    if cfg.get('backbone','TransE')=='RotatE':
        import rotate_api
        return rotate_api.active(include_attack)
    return official_api.active(include_attack)

def args_for(cfg,data,phase,seed,device):
    a=rt.build_args(cfg,'.',data.get('name','fb15k-237-10'),seed,device)
    a.snapshot=a.snapshot_valid=a.snapshot_test=phase;a.lr=cfg['delete_lr']
    a.unlearning_method=cfg['processor'];a.begin_pretrain=False;a.begin_unleanring=True;a.epoch=0
    return a

def make_model(up,args,kind,cfg):
    kg=cfg['_kg']
    if kind in ('P','R'):cls=up.Base if cfg.get('backbone')=='RotatE' else up.TransE
    elif cfg['processor']=='GraphDPO':
        import objective
        cls=up.GraphDPO if cfg['objective']=='code' else objective.pairwise_class(up.GraphDPO)
    else:cls=up.Plain if cfg.get('backbone')=='RotatE' else up.Repretrain
    return cls(args,kg).to(args.device)

def repair_budget(retained,scale=(1,1)):
    batches=(2*retained+1023)//1024;n,d=scale
    return (2*238*batches*n+100*d)//(200*d)

def stage_repair_steps(cfg,retained):
    if cfg.get('backbone')=='RotatE':
        import rotate_budgets
        arm='M_repair05' if cfg.get('repair_variant')=='half' else 'M'
        return rotate_budgets.repair_budget(arm,retained)
    return repair_budget(retained,cfg['repair_scale'])

def diagnostic_steps(cfg,data,smoke=False):
    if smoke:return 2
    if cfg.get('backbone')=='RotatE':
        import rotate_budgets
        return rotate_budgets.budget(len(gd.at(data,3)['retained']))
    if data['dataset']=='YA':return stage_repair_steps(cfg,len(gd.at(data,3)['retained']))
    return {'FB':1010,'WN':610,'CL':1709}[data['dataset']]

def snapshot(model,opt,metadata,gen=None):return rt.snapshot_checkpoint(model,opt,metadata,gen)

def pretrain(data,cfg,seed,device,phase=None):
    import types
    up,_=model_api(cfg);rt.seed_rng(seed);a=args_for(cfg,data,phase or 0,seed,device)
    facts=data['triples'] if phase is None else gd.at(data,phase)['retained']
    kg=types.SimpleNamespace(ent_num=data['counts']['entities'],rel_num=2*data['counts']['relations'],train_data=facts)
    cls=up.Base if cfg.get('backbone')=='RotatE' else up.TransE
    model=cls(a,kg).to(device);opt=rt.new_optimizer(model,cfg['pretrain_lr'])
    rt.seed_rng(rt.schedule_seed(seed,-1));ds=up.TrainDataset(a,kg);loader,gen=rt._loader(ds,1024,rt.schedule_seed(seed,-1))
    steps=0
    for epoch in range(cfg['pretrain_epochs']):
        for batch in loader:rt.train_step(model,opt,batch,'pretrain',device);steps+=1
    return snapshot(model,opt,dict(kind='P' if phase is None else 'R',phase=phase,seed=seed,updates=steps),gen)

def delete(data,cfg,seed,phase,device,start,previous_delete=None):
    up,_=model_api(cfg,cfg['processor']=='ADVIMP');a=args_for(cfg,data,phase,seed,device)
    rt.seed_rng(seed);model=make_model(up,a,'delete',dict(cfg,_kg=gd.kg(data,phase)))
    rt.load_model_state(model,start['model']);opt=rt.new_optimizer(model,a.lr)
    if previous_delete is not None:
        opt.load_state_dict(previous_delete['optimizer']);rt.restore_rng(previous_delete['rng'])
    initial=rt.tensor_tree_hash(model.state_dict());initial_opt=rt.tensor_tree_hash(opt.state_dict())
    graph=cfg['processor']=='GraphDPO'
    if graph:model.save_embeddings()
    teacher=rt.tensor_tree_hash(model.old_embeddings()) if graph else None
    rt.seed_rng(rt.schedule_seed(seed,phase))
    if cfg['fisher_stream']:
        import fisher_stream
        fisher_stream.install(up,64)
    push=cfg.get('push_only',False)
    if push:
        import no_pos_data
        proc=no_pos_data.processor(up,a,gd.kg(data,phase))
    else:
        proc=tk.make_processor(up,a,gd.kg(data,phase),model,opt,'GraphDPO_B' if graph else cfg['processor'])
    gens={}
    for name,value in vars(proc).items():
        if isinstance(value,torch.utils.data.DataLoader) and value.generator is not None:
            value.generator.manual_seed(rt.schedule_seed(seed,phase));gens[name]=value.generator
    receipt=None;weights=None;original=None;actual=None
    if graph:
        if not push:original=dict(facts=list(proc.dataset.pos_facts[::2]),is_heads=data['is_heads'][phase])
        model.entity_distill_mask=proc.entity_distill_mask
        if cfg['real_targets'] and not push:receipt=tk.t_targets(data,phase,seed,proc,igpu_only=True)
        actual=None if push else list(proc.dataset.pos_facts[::2])
        if cfg['gated']:
            qs,raw,s3,q,w=features.compute(data,phase)
            if cfg['gate'] in ('degree','shuffled'):
                import gate_features
                w=gate_features.variants(gd.at(data,phase)['retained'],qs,w,phase)[0][cfg['gate']]
            weights=dict(zip(map(tuple,qs.tolist()),w.tolist()))
    native=cfg.get('native_stop',False);selector=None;best=None;events=[]
    if native:
        from baselines.native_stop import NativeStop
        from baselines.native_validation import transformed_epoch
        a.valid_gap=3;a.valid_metrics='mrr';a.patience=1;a.epoch_num=100
        kg=gd.kg(data,phase);kg.hr2t=gd.filters(data['triples'])['legacy']
        dev=[up.ForgetDevBatchProcessor(a,kg),up.ReserveDevBatchProcessor(a,kg)]
        if data['dataset']=='WN':
            for d in dev:d.process_epoch=transformed_epoch(up,d)
        selector=NativeStop()
    count=[0];hook=opt.register_step_post_hook(lambda *unused:count.__setitem__(0,count[0]+1))
    trace=[];gradient=[];main=boundary=0;batch_digest=hashlib.sha256()
    try:
        for epoch in range(cfg['delete_epochs']):
            a.epoch=epoch
            if native:a.valid=True
            if cfg['gated']:
                go=gate_sensitivity if cfg['gate_scale']>1 else gate_core;losses=[]
                if go is gate_sensitivity:go.MAX_WEIGHT=8 if cfg.get('backbone')=='RotatE' else 4
                for batch in proc.data_loader:
                    ws=[cfg['gate_scale']*(1. if cfg['gate']=='uniform' else weights[tuple(q)]) for q in batch[3 if push else 5].tolist()]
                    batch_digest.update(b'main/');batch_digest.update(rt.tensor_tree_hash(batch).encode())
                    if push:
                        import no_pos_core
                        value,diag=no_pos_core.step(model,opt,batch,ws,rt,device)
                    else:value,diag=go.step(model,opt,batch,ws,rt,device,unit_check=main==0)
                    main+=1;losses.append(value['total']);gradient.append(dict(step=count[0],**diag))
                for batch in proc.forget_boundary_dataloader:
                    batch_digest.update(b'replay/');batch_digest.update(rt.tensor_tree_hash(batch).encode())
                    rt.replay_step(model,opt,batch,device);boundary+=1
                loss=math.fsum(losses)
            else:loss=proc.process_epoch(model,opt,proc.origin_params) if cfg['processor']=='ADVIMP' else proc.process_epoch(model,opt)
            if not math.isfinite(float(loss)) or not all(bool(torch.isfinite(p).all()) for p in model.parameters()):raise FloatingPointError('Nonfinite official result; preserve failure, no automatic repair')
            trace.append(dict(epoch=epoch+1,optimizer_steps=count[0],loss=float(loss)))
            if native and selector.due(epoch):
                fd,rd=(d.process_epoch(model) for d in dev)
                a.valid=False
                event=selector.observe(epoch,fd,rd);events.append(event)
                if event['improved']:best=rt.cpu_tree(model.state_dict())
                if event['stop']:break
    finally:hook.remove()
    if native and best is not None:rt.load_model_state(model,best)
    if graph and rt.tensor_tree_hash(model.old_embeddings())!=teacher:raise ValueError('Rolling teacher changed')
    payload=snapshot(model,opt,dict(kind='delete',seed=seed,phase=phase,updates=count[0],
        initial_model_tree=initial,initial_delete_Adam_tree=initial_opt,rolling_teacher_tree=teacher,
        loader_states={n:g.get_state().cpu() for n,g in gens.items()}),gens.get('data_loader'))
    return payload,dict(trace=trace,gradient=gradient,target_receipt=receipt,original_preferred=original,actual_positives=actual,
                        main_steps=main,boundary_steps=boundary,batch_schedule_sha256=batch_digest.hexdigest(),
                        native_validation=events,selected_epoch=selector.best_epoch if native else None,
                        stopped_epoch=trace[-1]['epoch'] if native else None)

def repair(data,cfg,seed,phase,device,source,steps):
    up,_=model_api(cfg);a=args_for(cfg,data,phase,seed,device)
    model=make_model(up,a,'delete',dict(cfg,_kg=gd.kg(data,phase)));rt.load_model_state(model,source['model'])
    opt=rt.new_optimizer(model,cfg['repair_lr']);rt.seed_rng(seed);a.unlearning_method='Finetune'
    ds=up.RepreTrainDataset(a,gd.kg(data,phase));loader,gen=rt._loader(ds,1024,seed)
    done=0;digest=hashlib.sha256()
    while done<steps:
        for batch in loader:
            digest.update(rt.tensor_tree_hash(batch).encode());rt.train_step(model,opt,batch,'pretrain',device);done+=1
            if done==steps:break
    return snapshot(model,opt,dict(kind='repair',seed=seed,phase=phase,updates=done,schedule_sha256=digest.hexdigest(),new_empty_Adam=True),gen)

def load_model(data,cfg,seed,phase,device,payload):
    up,_=model_api(cfg);a=args_for(cfg,data,phase,seed,device)
    kind='P' if payload['metadata']['kind'] in ('P','R') else 'delete'
    m=make_model(up,a,kind,dict(cfg,_kg=gd.kg(data,phase)));rt.load_model_state(m,payload['model']);m.eval()
    for p in m.parameters():p.requires_grad_(False)
    return m
