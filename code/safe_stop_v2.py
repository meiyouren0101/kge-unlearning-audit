"""User-invoked Linux-only v2 stop. Never deletes files or signals foreign jobs."""
import argparse, contextlib, json, os, select, signal, sys, time, uuid
from pathlib import Path
from datetime import datetime, timezone

SHARED_LOCK=Path('/mnt/new4T/lsb/.graphdpo_gpu0_admission.lock')
WAITING={'WAIT_GPU','WAIT_GPU_LOCK','WAIT_DEPENDENCY','QUEUED'}
SCRIPTS={'job20.py','worker20.py'}

def require(value,message):
    if not value:raise ValueError(message)

def read(path):return json.loads(Path(path).read_text('utf-8'))

def process_record(pid,proc_root=Path('/proc')):
    folder=Path(proc_root)/str(pid)
    try:
        fields=(folder/'stat').read_text().rpartition(') ')[2].split()
        if not fields or fields[0]=='Z':return None
        cmd=[x.decode('utf-8','replace') for x in (folder/'cmdline').read_bytes().split(b'\0') if x]
        cwd=(folder/'cwd').resolve(strict=True)
        return dict(pid=int(pid),ppid=int(fields[1]),start_ticks=int(fields[19]),uid=folder.stat().st_uid,cmd=cmd,cwd=str(cwd))
    except (FileNotFoundError,ProcessLookupError,PermissionError):return None

def resolved_script(record,root):
    root=Path(root).resolve()
    for token in record['cmd'][1:]:
        p=Path(token)
        if p.name not in SCRIPTS:continue
        target=(p if p.is_absolute() else Path(record['cwd'])/p).resolve()
        if target.parent==root:return target.name
    return None

def matching_processes(root,proc_root=Path('/proc')):
    found=[]
    for p in Path(proc_root).iterdir():
        if not p.name.isdigit():continue
        r=process_record(int(p.name),proc_root)
        if r is not None:
            script=resolved_script(r,root)
            if script:found.append(dict(r,script=script))
    return sorted(found,key=lambda r:r['pid'])

def descendants(pid,proc_root=Path('/proc')):
    rows=[]
    for p in Path(proc_root).iterdir():
        if p.name.isdigit():
            r=process_record(int(p.name),proc_root)
            if r is not None:rows.append(r)
    active={pid};result=[]
    while True:
        new=[r for r in rows if r['pid'] not in active and r['ppid'] in active]
        if not new:break
        result+=new;active.update(r['pid'] for r in new)
    return result

def snapshot(root):
    root=Path(root).resolve(strict=True)
    require(root.name=='graphdpo_fb_wn20_full_baselines_v2_job','Only the exact v2 package is eligible')
    session=Path(read(root/'latest_run.json')['session']).resolve(strict=True)
    require(session.is_relative_to(root/'results'),'v2 pointer outside its own results')
    states={};counts={}
    for tag in ('FB','WN'):
        s=session/tag;p=s/'gpu_wait_state.json'
        require(s.is_dir() and p.is_file(),'Missing v2 chain state '+tag)
        states[tag]=read(p);counts[tag]=len(list((s/'stages').glob('*/commit.json')))
    processes=matching_processes(root)
    return dict(root=str(root),session=str(session),committed=counts,states=states,processes=processes)

def validate_waiting(s,uid,children):
    require(sum(s['committed'].values())==0,'v2 has sealed stages; refuse this zero-stage switch')
    require(all(p['uid']==uid for p in s['processes']),'Process owner mismatch')
    require(not any(p['script']=='worker20.py' for p in s['processes']),'v2 worker is running; refuse to stop')
    coordinators=[p for p in s['processes'] if p['script']=='job20.py']
    require(len(coordinators)<=1,'Multiple v2 processes; inspect manually')
    if coordinators:
        require('run' in coordinators[0]['cmd'],'Not the v2 run coordinator')
        require(not children,'v2 coordinator has live children; refuse to signal')
        require(all(v.get('status') in WAITING for v in s['states'].values()),'v2 is not confirmed waiting')
    return coordinators[0] if coordinators else None

@contextlib.contextmanager
def admission_guard(path=SHARED_LOCK,timeout=120):
    import fcntl
    # Do not unlink or replace the shared lock inode, ever.
    with Path(path).open('a+b') as f:
        deadline=time.monotonic()+timeout
        while True:
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if time.monotonic()>=deadline:raise TimeoutError('Shared startup lock busy; no process signalled')
                time.sleep(1)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)

def coordinator_unlocked(root):
    import fcntl
    path=Path(root)/'.coordinator.lock'
    require(path.is_file(),'Missing v2 coordinator lock; inspect manually')
    with path.open('rb') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(f,fcntl.LOCK_UN)

def stop_waiting(root):
    require(sys.platform=='linux','Run this helper on the Linux server, not locally')
    require(hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'),'pidfd support required for PID-safe stop')
    with admission_guard():
        # Holding the shared start lock closes the check/start race with v2.
        s=snapshot(root)
        cs=[p for p in s['processes'] if p['script']=='job20.py']
        child=descendants(cs[0]['pid']) if len(cs)==1 else []
        target=validate_waiting(s,os.getuid(),child)
        if target:
            fd=os.pidfd_open(target['pid'],0)
            try:
                current=process_record(target['pid'])
                require(current is not None and current['start_ticks']==target['start_ticks'] and current['uid']==os.getuid() and resolved_script(current,root)=='job20.py','PID identity changed; no signal sent')
                print('SIGTERM_V2_COORDINATOR_ONLY='+str(target['pid']),flush=True)
                signal.pidfd_send_signal(fd,signal.SIGTERM,None,0)
                poll=select.poll();poll.register(fd,select.POLLIN)
                require(bool(poll.poll(90000)),'v2 has not exited within90s; no SIGKILL issued, inspect its log')
            finally:os.close(fd)
        after=snapshot(root)
        require(not after['processes'],'v2 process still alive; do not launch v3')
        require(sum(after['committed'].values())==0,'A stage sealed during switch; preserve it and inspect before v3')
        coordinator_unlocked(root)
    receipt=dict(status='V2_STOP_CONFIRMED',zero_sealed_stages=True,old_session=s['session'],signalled_pid=target['pid'] if target else None,
        no_remaining_v2_process=True,coordinator_lock_released=True,shared_start_lock_released=True,old_files_not_deleted=True,
        created_utc=datetime.now(timezone.utc).isoformat())
    folder=Path(__file__).resolve().parent/'switch_receipts';folder.mkdir(exist_ok=True)
    name=uuid.uuid4().hex;temp=folder/(name+'.tmp');out=folder/(name+'.json')
    temp.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');temp.replace(out)
    print('V2_STOP_CONFIRMED=True; remaining_v2_processes=0; sealed=0; locks_released=True',flush=True)
    print('RECEIPT='+str(out),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--v2-root',type=Path,required=True)
    p.add_argument('--stop',action='store_true');args=p.parse_args()
    require(sys.platform=='linux','Linux only')
    if args.stop:stop_waiting(args.v2_root)
    else:print(json.dumps(snapshot(args.v2_root),indent=2))

if __name__=='__main__':main()
