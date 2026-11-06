"""Shared GPU start lock, measured headroom, heartbeats and bounded OOM retry."""
from pathlib import Path
import csv
import math
import subprocess
import threading
import time
import common as c
import io_utils as io
MIN_FREE_MIB=4096
POLL_SECONDS=60
HEARTBEAT_SECONDS=600
MONITOR_SECONDS=5
START_GRACE_SECONDS=45
MAX_OOM_RETRIES=3
DEFAULT_STAGE_SECONDS=7200
FISHER_TIMEOUTS=(21600,43200,86400)
BASELINE_ARMS=frozenset(('Finetune','FT-lr1e-4','RL','NG','Fisher','BS','ADV-IMP','SSD','GraphDPO','GraphDPO-code','ZF','Schema'))
SHARED_LOCK='/mnt/new4T/lsb/.graphdpo_gpu0_admission.lock'
START_LEASES=threading.local()

class StopRequested(Exception):
    pass

def check_stop(stop):
    if stop is not None and stop.is_set():
        raise StopRequested('Audit cancelled after a coordinator or peer-chain failure')

def pause(seconds,stop=None):
    if stop is None:
        time.sleep(seconds)
    else:
        stop.wait(seconds)
        check_stop(stop)

def pending_leases():
    if not hasattr(START_LEASES,'pending'):
        START_LEASES.pending=[]
    return START_LEASES.pending

def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())

def query():
    def rows(fields,kind):
        r=subprocess.run(['nvidia-smi','--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=20)
        return [r for r in csv.reader(r.stdout.splitlines(),skipinitialspace=True) if r]
    device=[r for r in rows('index,uuid,memory.total,memory.free','gpu') if r[0].strip()=='0']
    io.require(len(device)==1 and len(device[0])==4,'GPU0 memory telemetry unavailable')
    index,uuid,total,free=[v.strip() for v in device[0]];total,free=int(total),int(free)
    io.require(0<=free<=total,'Invalid GPU memory telemetry');processes=[]
    for r in rows('gpu_uuid,pid,used_gpu_memory','compute-apps'):
        if r[0].strip()!=uuid:continue
        io.require(len(r)==3 and r[1].strip().isdigit(),'Invalid GPU0 process telemetry')
        value=r[2].strip();processes.append(dict(pid=int(r[1]),used_mib=int(value) if value.isdigit() else None))
    return dict(index=0,uuid=uuid,total_mib=total,free_mib=free,compute=processes)

def foreign(s,pid=None):return [p for p in s['compute'] if p['pid']!=pid]

def write_state(path,heartbeat=False,**values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);old=io.read(path) if path.exists() else {}
    now=utc();value=dict(old,updated_utc=now,**values)
    if 'snapshot' in values:value['gpu_sample_utc']=now
    if values.get('status')=='STAGE_START':value['min_free_mib']=None
    if values.get('status') in ('RUNNING','STARTING','COMPLETE') and 'reason' not in values:value['reason']=None
    if heartbeat or 'last_heartbeat_utc' not in value:value['last_heartbeat_utc']=now
    tmp=path.with_name('.'+path.name+'.'+io.stamp()+'.tmp');c.write(tmp,value);tmp.replace(path)

class StartLease:
    def __init__(self,path):self.path=Path(path);self.file=None;self.started_at=None;self.timer=None;self.guard=threading.Lock()
    def acquire(self):
        import fcntl
        self.path.parent.mkdir(parents=True,exist_ok=True);self.file=self.path.open('a+b')
        try:fcntl.flock(self.file,fcntl.LOCK_EX|fcntl.LOCK_NB);return True
        except BlockingIOError:self.file.close();self.file=None;return False
    def started(self,seconds=START_GRACE_SECONDS):
        pending=pending_leases()
        pending[:]=[lease for lease in pending if lease.file is not None];pending.append(self)
        self.started_at=time.monotonic();self.timer=threading.Timer(seconds,self.release);self.timer.daemon=True;self.timer.start()
    def release(self):
        import fcntl
        with self.guard:
            if self.file is not None:
                f=self.file;self.file=None;fcntl.flock(f,fcntl.LOCK_UN);f.close()

def release_before_dependency(path,stage):
    pending=pending_leases()
    for lease in pending:
        if lease.file is None:continue
        remaining=max(0,START_GRACE_SECONDS-(time.monotonic()-lease.started_at))
        if remaining:
            write_state(path,heartbeat=True,status='WAIT_GPU_LOCK',stage=stage,reason='FINISH_PRIOR_45S_START_GRACE_BEFORE_DEPENDENCY')
            time.sleep(remaining)
        lease.release()
    pending[:]=[lease for lease in pending if lease.file is not None]

class ResourceLimit(Exception):pass

def admission_threshold(peak,total):
    if peak is None:value=MIN_FREE_MIB
    else:
        value=math.ceil(1.5*peak)
        if value>total:value=math.ceil(peak+1024)
    if value>total:raise ResourceLimit('Measured headroom exceeds full GPU: '+str(value)+'>'+str(total)+' MiB')
    return value

def wait_for_gpu(path,stage,min_free_mib=MIN_FREE_MIB,poll_seconds=POLL_SECONDS,lock_path=SHARED_LOCK,measured_peak=None,stop=None):
    polls=0;last_heartbeat=None
    while True:
        check_stop(stop)
        lease=StartLease(lock_path) if lock_path else None
        locked=lease is None or lease.acquire()
        if not locked:
            beat=last_heartbeat is None or time.monotonic()-last_heartbeat>=HEARTBEAT_SECONDS
            write_state(path,heartbeat=beat,status='WAIT_GPU_LOCK',stage=stage,reason='SHARED_START_LOCK')
            if beat:print('HEARTBEAT '+utc()+': WAIT_GPU_LOCK '+stage,flush=True);last_heartbeat=time.monotonic()
            pause(poll_seconds,stop);continue
        try:
            s=query()
            required=max(min_free_mib,admission_threshold(measured_peak,s['total_mib']))
            if required>s['total_mib']:raise ResourceLimit('Admission threshold exceeds total VRAM')
        except BaseException:
            if locked and lease is not None:lease.release()
            raise
        enough=s['free_mib']>=required
        if locked and enough:
            write_state(path,heartbeat=True,status='STARTING',stage=stage,wait_polls=polls,min_free_mib=required,snapshot=s)
            print('GPU_READY '+utc()+': '+stage+'; free='+str(s['free_mib'])+'MiB; automatic start under shared lock',flush=True)
            return dict(s,_lease=lease)
        if locked and lease is not None:lease.release()
        polls+=1;reason='SHARED_START_LOCK' if not locked else 'INSUFFICIENT_FREE_VRAM'
        beat=last_heartbeat is None or time.monotonic()-last_heartbeat>=HEARTBEAT_SECONDS
        write_state(path,heartbeat=beat,status='WAIT_GPU_LOCK' if not locked else 'WAIT_GPU',stage=stage,reason=reason,wait_polls=polls,min_free_mib=required,snapshot=s)
        if beat:
            print('HEARTBEAT '+utc()+': '+reason+'; stage='+stage+'; free='+str(s['free_mib'])+'/'+str(required)+'MiB; automatic retry in '+str(poll_seconds)+'s',flush=True);last_heartbeat=time.monotonic()
        pause(poll_seconds,stop)

class Watcher:
    def __init__(self,start,state_path=None,stage=None):
        self.start=dict(start);self.lease=self.start.pop('_lease',None);self.started_flag=False;self.max_used_mib=0
        self.competition_seen=bool(self.start['compute']);self.own_memory_seen=False
        self.events=[];self.state_path=state_path;self.stage=stage;self.last_heartbeat=time.monotonic()
    def started(self,pid):
        self.started_flag=True
        if self.lease is not None:self.lease.started()
        if self.state_path:write_state(self.state_path,heartbeat=True,status='RUNNING',stage=self.stage,worker_pid=pid)
    def observe(self,s,pid):
        own=[p['used_mib'] for p in s['compute'] if p['pid']==pid and p['used_mib'] is not None]
        if own:self.max_used_mib=max(self.max_used_mib,max(own));self.own_memory_seen=True
        external=s['total_mib']-s['free_mib']-(max(own) if own else 0)
        if foreign(s,pid) or own and external>self.start['total_mib']-self.start['free_mib']+256:self.competition_seen=True
        self.events.append(dict(snapshot=s,worker_pid=pid));self.events=self.events[-16:]
        if self.state_path:
            beat=time.monotonic()-self.last_heartbeat>=HEARTBEAT_SECONDS
            write_state(self.state_path,heartbeat=beat,status='RUNNING',stage=self.stage,snapshot=s,worker_pid=pid)
            if beat:print('HEARTBEAT '+utc()+': RUNNING; stage='+self.stage+'; free='+str(s['free_mib'])+'MiB',flush=True);self.last_heartbeat=time.monotonic()
    def sample(self,pid):self.observe(query(),pid)
    def abort_before_start(self):
        if self.lease is not None and not self.started_flag:self.lease.release()

def is_cuda_oom(exc):
    text=str(exc).lower();return not isinstance(exc,MemoryError) and 'cuda' in text and ('out of memory' in text or 'outofmemory' in text)

def torch_peak_mib():
    import sys
    torch=sys.modules.get('torch')
    return math.ceil(torch.cuda.max_memory_reserved(0)/1024**2) if torch is not None and torch.cuda.is_initialized() else None

def clear_stage_cache(out):
    import gc
    import sys
    gc.collect();torch=sys.modules.get('torch');initialized=bool(torch is not None and torch.cuda.is_initialized())
    try:
        if initialized:torch.cuda.empty_cache()
        c.write(Path(out)/'cuda_cleanup.json',dict(cuda_initialized=initialized,empty_cache_called=initialized,stage_process_exits_afterward=True))
    except BaseException as exc:
        c.write(Path(out)/'cleanup_failure.json',dict(type=type(exc).__name__,reason=str(exc)));raise

def oom_marker(folder,spec,exc,snapshot=None,telemetry_error=None):
    c.write(Path(folder)/'gpu_oom.json',dict(status='GPU_OOM',stage=spec['name'],mode=spec.get('mode'),arm=spec.get('arm'),seed=spec.get('seed'),phase=spec.get('phase'),
        type=type(exc).__name__,reason=str(exc),snapshot=snapshot,telemetry_error=telemetry_error,torch_peak_reserved_mib=torch_peak_mib(),optimizer_state_never_committed=True))

def peak_key(session,spec):return '/'.join((session.name[:2],spec['mode'],spec.get('arm','fixtures')))

def remember_peak(path,key,peak,completed=True):
    if not isinstance(peak,(int,float)) or peak<=0:return
    old=io.read(path) if path.exists() else {};entry=old.get(key,{})
    if isinstance(entry,(int,float)):entry=dict(peak_mib=entry,completed=True)
    old[key]=dict(peak_mib=max(entry.get('peak_mib',0),peak),completed=bool(entry.get('completed') or completed))
    tmp=path.with_name('.'+path.name+'.'+io.stamp()+'.tmp');c.write(tmp,old);tmp.replace(path)

def required_memory(path,key):
    values=io.read(path) if path.exists() else {};entry=values.get(key)
    if entry is None:return MIN_FREE_MIB
    if isinstance(entry,(int,float)):return math.ceil(entry*1.5)
    measured=math.ceil(entry['peak_mib']*1.5)
    return measured if entry['completed'] else max(MIN_FREE_MIB,measured)

def archive_retry(folder,log,session,spec,number,reason,watcher):
    folder,session,log=Path(folder),Path(session),Path(log)
    io.require(folder.resolve().parent==session.resolve()/'stages' and not folder.is_symlink(),'Retry stage out of scope')
    target=session/'resource_retries'/spec['name']/('attempt_'+str(number).zfill(3));target.mkdir(parents=True,exist_ok=False)
    io.require(target.resolve().is_relative_to(session.resolve()),'Retry archive out of scope');folder.rename(target/'stage')
    if log.exists():log.rename(target/'worker.log')
    hashes={p.relative_to(target/'stage').as_posix():io.sha(p) for p in (target/'stage').rglob('*') if p.is_file()}
    c.write(target/'retry.json',dict(stage=spec['name'],attempt=number,reason=reason,evidence=watcher.events,failed_files_sha256=hashes,
        identical_stage_spec=spec,restart='same stage from its original committed model/Adam/RNG inputs',partial_outputs_not_used=True,deterministic_bridge_required=True))
    folder.mkdir();print('GPU_RESOURCE_RETRY '+utc()+': '+spec['name']+'; preserved='+str(target)+'; archived failed attempt '+str(number),flush=True)

def own_baseline_oom(spec,watcher):
    # Session names are FB / WN. Competition is sticky over the entire attempt.
    return spec.get('arm') in BASELINE_ARMS and watcher.own_memory_seen and not watcher.competition_seen

def stage_timeout(spec,timeouts=0,default=DEFAULT_STAGE_SECONDS):
    if spec.get('mode')=='delete' and spec.get('arm')=='Fisher':
        io.require(0<=timeouts<len(FISHER_TIMEOUTS),'Fisher timeout budget exhausted')
        return FISHER_TIMEOUTS[timeouts]
    return default

def terminal_resource(folder,spec,reason,**extra):
    if (folder/'result.json').exists():(folder/'result.json').rename(folder/'partial_result_before_resource.json')
    row=dict(spec,status='RESOURCE_LIMIT',reason=reason,residue_preserved=True,**extra)
    # Protected scientific references / main method are never silently skipped.
    if spec.get('arm') in ('P','R','IGPU'):
        c.write(folder/'resource_failure.json',row)
        raise ResourceLimit('Protected '+spec['arm']+': '+reason)
    c.write(folder/'result.json',row)

def run_stage(cmd,folder,log,session,spec,invoke,timeout=DEFAULT_STAGE_SECONDS,poll_seconds=POLL_SECONDS,lock_path=SHARED_LOCK,stop=None):
    folder,session=Path(folder),Path(session);state_path=session/'gpu_wait_state.json';peaks=session/'gpu_peak_history.json';key=peak_key(session,spec)
    prior=sorted((session/'resource_retries'/spec['name']).glob('attempt_*'))
    io.require([p.name for p in prior]==['attempt_'+str(k).zfill(3) for k in range(1,len(prior)+1)],'Invalid retry archive sequence')
    reasons=[io.read(p/'retry.json')['reason'] for p in prior]
    io.require(all(r in ('CUDA OOM','WALL_TIMEOUT') for r in reasons),'Unknown retry reason')
    oom_count=reasons.count('CUDA OOM');timeout_count=reasons.count('WALL_TIMEOUT');attempt=len(prior)
    io.require(oom_count<=MAX_OOM_RETRIES and timeout_count<3,'Invalid retry budgets')
    while True:
        check_stop(stop)
        registry=io.read(peaks) if peaks.exists() else {};entry=registry.get(key)
        peak=entry.get('peak_mib') if isinstance(entry,dict) else entry
        initial=max(MIN_FREE_MIB,math.ceil(1.5*spec['estimated_peak_mib']))
        floor=0 if isinstance(entry,dict) and entry['completed'] else initial
        limit=stage_timeout(spec,timeout_count,timeout or DEFAULT_STAGE_SECONDS)
        write_state(state_path,stage=session.name+'/'+spec['name'],timeout_seconds=limit,oom_count=oom_count,timeout_count=timeout_count)
        try:start=wait_for_gpu(state_path,session.name+'/'+spec['name'],floor,poll_seconds,lock_path,measured_peak=peak,stop=stop)
        except ResourceLimit as exc:
            terminal_resource(folder,spec,str(exc));return 0,False,None
        watcher=Watcher(start,state_path,session.name+'/'+spec['name'])
        began=time.monotonic()
        try:code,timed,pid=invoke(cmd,log,timeout=limit,cpu=False,monitor=watcher)
        finally:watcher.abort_before_start()
        io.require(not (folder/'cleanup_failure.json').exists(),'CUDA stage cleanup failed; stop and preserve')
        reason=None;marker=folder/'gpu_oom.json'
        if marker.exists():
            r=io.read(marker);io.require(r['status']=='GPU_OOM' and r['stage']==spec['name'],'GPU OOM marker identity')
            if r.get('snapshot'):watcher.observe(r['snapshot'],pid)
            peak=max(watcher.max_used_mib,r.get('torch_peak_reserved_mib') or 0);remember_peak(peaks,key,peak,completed=False)
            oom_count+=1
            if own_baseline_oom(spec,watcher):
                terminal_resource(folder,spec,'Baseline CUDA OOM without observed competing compute or external VRAM growth',automatic_retry=False,oom_count=oom_count,timeout_count=timeout_count)
                return 0,False,pid
            if oom_count>MAX_OOM_RETRIES:
                terminal_resource(folder,spec,'CUDA OOM after initial attempt and three retries',oom_count=oom_count,timeout_count=timeout_count)
                return 0,False,pid
            reason='CUDA OOM'
        elif timed:
            timeout_count+=1
            c.write(folder/'wall_timeout.json',dict(limit_seconds=limit,observed_seconds=time.monotonic()-began,worker_pid=pid,timeout_count=timeout_count,oom_count=oom_count,only_own_worker_stopped=True))
            if spec.get('mode')!='delete' or spec.get('arm')!='Fisher' or timeout_count==3:
                terminal_resource(folder,spec,'Wall-clock stage limit reached',timeout_seconds=limit,timeout_count=timeout_count,oom_count=oom_count)
                return 0,False,pid
            reason='WALL_TIMEOUT'
        if reason:
            attempt+=1;archive_retry(folder,log,session,spec,attempt,reason,watcher)
            # Completed worker owns no GPU allocation; unlock before retry cooldown.
            if watcher.lease is not None:watcher.lease.release()
            write_state(state_path,heartbeat=True,status='WAIT_GPU',stage=session.name+'/'+spec['name'],reason=reason+'_COOLDOWN',oom_count=oom_count,timeout_count=timeout_count)
            pause(poll_seconds,stop);continue
        if code==0 and (folder/'result.json').exists():
            r=io.read(folder/'result.json');remember_peak(peaks,key,max(watcher.max_used_mib,r.get('gpu_peak_reserved_mib') or 0,r.get('gpu_peak_used_mib') or 0))
            if attempt:
                bridge=io.read(folder/'determinism_bridge.json');io.require(bridge['status']=='PASS' and bridge['retry_attempts']==attempt,'Retried result lacks deterministic bridge')
        return code,False,pid
