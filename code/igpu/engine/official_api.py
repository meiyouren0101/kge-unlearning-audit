"""Execute complete pinned official classes; source bytes are never patched."""
from pathlib import Path
import ast
import copy
import importlib
import math
import itertools
import os
import random
import sys
import types
import common as c
import io_utils as io

def advertorch_l2():
    """Load unmodified L2PGD and its dependencies, without unrelated package reexports.

    advertorch's all-attacks initializer imports unused FAB/private removed torch APIs.
    This import-scope adaptation executes the exact 0.2.3 L2PGD implementation used
    by GraphDPO, and does not replace an attack or a torch function.
    """
    root=c.ROOT_VENDOR()/'advertorch';receipt=io.read(c.ROOT/'vendor_receipt.json')
    for name,row in receipt['files'].items():io.require(io.sha(c.ROOT_VENDOR()/name)==row['sha256'],'Dependency source changed')
    for name,path in (('advertorch',root),('advertorch.attacks',root/'attacks')):
        if name not in sys.modules:
            module=types.ModuleType(name);module.__path__=[str(path)];module.__package__=name
            sys.modules[name]=module
    base=importlib.import_module('advertorch.attacks.base')
    pkg=sys.modules['advertorch.attacks'];pkg.Attack=base.Attack;pkg.LabelMixin=base.LabelMixin
    module=importlib.import_module('advertorch.attacks.iterative_projected_gradient')
    pkg.L2PGDAttack=module.L2PGDAttack
    return module.L2PGDAttack

def active(include_attack=False):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader,Dataset
    from tqdm import tqdm
    c.official_check()
    ns={'__name__':'graphdpo_official_active','__builtins__':__builtins__,'torch':torch,'nn':nn,'F':F,
        'np':np,'math':math,'os':os,'random':random,'itertools':itertools,'deepcopy':copy.deepcopy,'Dataset':Dataset,'DataLoader':DataLoader,
        'Dict':dict,'List':list,'xavier_normal_':nn.init.xavier_normal_,
        'GradScaler':torch.cuda.amp.GradScaler,'autocast':torch.cuda.amp.autocast,'tqdm':tqdm}
    if include_attack:ns['L2PGDAttack']=advertorch_l2()
    files=(('src/model/TransE.py',('TransE',)),('src/model/Repretrain.py',('Repretrain',)),
        ('src/model/GraphDPO.py',('GraphDPO',)),('src/data_load/data_loader.py',None),
        ('src/model/model_process.py',None),('src/train.py',('UnlearningTrainer',)))
    records={}
    for name,names in files:
        path=c.ROOT_OFFICIAL()/name;tree=ast.parse(path.read_text('utf-8'))
        nodes=[x for x in tree.body if isinstance(x,ast.ClassDef) and (names is None or x.name in names)]
        if names is not None:io.require(len(nodes)==len(names),'Required original class missing')
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
        records[name]=io.sha(path)
    return types.SimpleNamespace(**ns),records
