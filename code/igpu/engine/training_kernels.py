"""Verbatim processor selection and retained-target construction."""
import copy
import common as c
import graph_data as gd
import legacy_runtime as rt
import io_utils as io

def make_processor(up,args,kg,model,opt,arm):
    names=dict(Finetune='TrainBatchProcessor',**{'FT_lr1e-4':'TrainBatchProcessor'},RL='RLTrainBatchProcessor',NG='NGTrainBatchProcessor',Fisher='FisherTrainBatchProcessor',BS='BSTrainBatchProcessor',ADVIMP='ADVIMPTrainBatchProcessor',SSD='SSDTrainBatchProcessor')
    cls=getattr(up,'GraphDPOTrainBatchProcessor' if arm in c.GRAPH_ARMS else names[arm])
    if arm=='ADVIMP':
        proc=cls(args,kg,model);proc.origin_params={n:p.clone().detach() for n,p in model.named_parameters() if p.requires_grad}
        proc.importance=up.UnlearningTrainer.estimate_parameter_importance(None,proc.data_loader,copy.deepcopy(model),args.device,kg.snapshots[args.snapshot].forgotten_num,opt)
    else:proc=cls(args,kg)
    return proc

def t_targets(data,phase,seed,proc,igpu_only=False):
    import target_plan as tp
    import full_plan as fp
    omitted=None
    if igpu_only:
        import igpu_filter
        d,omitted=igpu_filter.apply(data,phase,proc)
    else:
        d=gd.at(data,phase);d={**d,'phases':[[],[],data['phases'][phase]],'is_heads':[[],[],data['is_heads'][phase]]}
    original=dict(positive=proc.dataset.pos_facts,negative=proc.dataset.neg_facts,facts=proc.dataset.facts,boundary=proc.forget_boundary_dataset.facts,
        legacy_dataset_sha256=rt.tensor_tree_hash(proc.dataset.pos_facts))
    plan=tp.TargetPlan(d,original,seed);targets,positions,receipt=fp.build(d,plan,seed)
    fp.apply(proc.dataset,targets,d['phases'][2])
    if omitted is not None:receipt['igpu_empty_positive_policy']=omitted
    return receipt
