"""Exact deterministic inputs/update prefixes and frozen rank prefixes on retry."""
from pathlib import Path
import hashlib
import json
import os
import common as c
import io_utils as io
ACTIVE=None

def record(kind,index,model,optimizer=None):
    if ACTIVE is not None:ACTIVE.tensor_anchor(kind,index,model,optimizer)

class Bridge:
    def __init__(self,session,spec,out):
        self.session,self.spec,self.out=Path(session),spec,Path(out);self.expected={};self.seen=set();self.archives=[];self.count=0
        for folder in sorted((self.session/'resource_retries'/spec['name']).glob('attempt_*')):
            receipt=io.read(folder/'retry.json')
            for n,h in receipt['failed_files_sha256'].items():io.require(io.sha(io.child(folder/'stage',n))==h,'Failed-attempt evidence changed')
            self.archives.append(folder/'stage');path=folder/'stage/determinism_anchors.jsonl'
            if path.exists():
                for line in path.read_text('utf-8').splitlines():
                    r=json.loads(line);key=r['key'];value=r['sha256']
                    io.require(key not in self.expected or self.expected[key]==value,'Previous retries disagree on deterministic prefix');self.expected[key]=value
        self.handle=(self.out/'determinism_anchors.jsonl').open('x',encoding='utf-8',newline='\n');self.original_optimizer_factory=None;self.hooks=[]
    def anchor(self,key,digest):
        if key in self.expected:io.require(digest==self.expected[key],'Deterministic retry prefix mismatch: '+key)
        self.seen.add(key);self.handle.write(json.dumps(dict(key=key,sha256=digest),sort_keys=True)+'\n');self.handle.flush();os.fsync(self.handle.fileno())
    def tensor_anchor(self,kind,index,model,optimizer=None):
        import legacy_runtime as rt
        values=dict(model=model.state_dict(),rng=rt.capture_rng())
        if optimizer is not None:values['optimizer']=optimizer.state_dict()
        self.anchor(kind+'/'+str(index),rt.tensor_tree_hash(values))
    def __enter__(self):
        try:return self.start()
        except BaseException:self.__exit__(None,None,None);raise
    def start(self):
        global ACTIVE
        io.require(ACTIVE is None,'Nested resource bridge');ACTIVE=self
        names=['protocol.json','plan.json','segment.json','historical.json','graph_inputs.json','asset_receipt.json','preferred_receipt.json','evaluation_samples.json']
        inputs={n:io.sha(self.session/n) for n in names if (self.session/n).is_file()}
        plan_path=self.session/'plan.json'
        if plan_path.exists():
            import dependencies
            plan=io.read(plan_path)
            required=dependencies.names(self.spec,plan)
            for name in required:
                pin=self.session/'stages'/name/'commit.json'
                io.require(pin.is_file(),'Unsealed retry input dependency '+name)
                inputs[pin.relative_to(self.session).as_posix()]=io.sha(pin)
        else:
            # Synthetic bridge tests have no real plan; production always does.
            io.require(not (self.session/'segment.json').exists(),'Missing runtime plan')
        value=json.dumps(dict(spec=self.spec,inputs=inputs),sort_keys=True,separators=(',',':')).encode();self.anchor('stage_inputs',hashlib.sha256(value).hexdigest())
        import legacy_runtime as rt
        self.original_optimizer_factory=rt.new_optimizer
        def factory(model,lr):
            opt=self.original_optimizer_factory(model,lr);self.count+=1;ordinal=self.count
            self.tensor_anchor('optimizer_construct_'+str(ordinal),0,model,opt);steps=[0]
            def before_step(*a,**kw):
                steps[0]+=1
                if steps[0] & (steps[0]-1)==0:self.tensor_anchor('optimizer_pre_step_'+str(ordinal),steps[0],model,opt)
            self.hooks.append(opt.register_step_pre_hook(before_step));return opt
        rt.new_optimizer=factory;return self
    def finish(self):
        io.require(set(self.expected)<=self.seen,'Retry did not replay every completed deterministic prefix');prefixes=[]
        for archive in self.archives:
            for old in archive.glob('*.u32le'):
                new=self.out/old.name;io.require(new.is_file() and new.stat().st_size>=old.stat().st_size,'Retry missing frozen rank prefix')
                with old.open('rb') as a,new.open('rb') as b:
                    for chunk in iter(lambda:a.read(1024**2),b''):io.require(b.read(len(chunk))==chunk,'Frozen rank retry prefix mismatch')
                prefixes.append(dict(artifact=old.name,bytes=old.stat().st_size,exact=True))
        r=dict(status='PASS' if self.archives else 'NOT_REQUIRED',retry_attempts=len(self.archives),verified_anchor_count=len(self.expected),new_anchor_count=len(self.seen),frozen_rank_prefixes=prefixes,
            scope='exact committed inputs and completed tensor/update prefixes; exact previously emitted ranks; no nonexistent failed endpoint is compared')
        c.write(self.out/'determinism_bridge.json',r);return r
    def __exit__(self,*args):
        global ACTIVE
        import legacy_runtime as rt
        if self.original_optimizer_factory is not None:rt.new_optimizer=self.original_optimizer_factory
        for hook in self.hooks:hook.remove()
        ACTIVE=None;self.handle.close()
