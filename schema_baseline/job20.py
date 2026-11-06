"""FB/WN 20% shared-GPU coordinator; never admits other jobs by PID."""
import argparse, contextlib, json, os, shutil, signal, subprocess, sys, time, traceback, uuid, zipfile
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0,str(Path(__file__).resolve().parent))
from igpu import bootstrap
import common as c, gpu_wait as gpu, matrix20 as m
from datetime import datetime,timezone
ROOT=m.ROOT
LOCK=Path('/mnt/new4T/lsb/.graphdpo_gpu0_admission.lock')
def stamp():return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
def atomic(p,v):
    q=p.with_name('.'+p.name+'.'+stamp()+'.tmp');c.write(q,v);q.replace(p)
def hashes():
    x=m.read(ROOT/'PACKAGE_SHA256.json')
    for p,h in x['files'].items():m.verify_file(ROOT/p,h)
def seal(out):
    m.check(not (out/'commit.json').exists(),'Duplicate stage seal')
    atomic(out/'commit.json',dict(status='COMMITTED',files=m.signed_payload(out)))
def stop_own_worker(p):
    if p.poll() is not None:return
    p.terminate()
    try:p.wait(timeout=30)
    except subprocess.TimeoutExpired:p.kill();p.wait()

def invoke(cmd,log,timeout=None,cpu=False,monitor=None,stop=None,poll_seconds=5):
    env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='' if cpu else '0')
    gpu.check_stop(stop)
    with Path(log).open('xb') as f:
        p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
        began=time.monotonic()
        try:
            if monitor:monitor.started(p.pid)
            while p.poll() is None:
                gpu.check_stop(stop)
                elapsed=time.monotonic()-began
                if timeout is not None and elapsed>=timeout:
                    stop_own_worker(p);return p.returncode,True,p.pid
                if monitor:monitor.sample(p.pid)
                gpu.pause(min(poll_seconds,max(.001,timeout-elapsed)) if timeout else poll_seconds,stop)
            return p.returncode,False,p.pid
        except BaseException:
            stop_own_worker(p);raise

def parallel_chains(tags,workers,action):
    stop=threading.Event();errors=[]
    def one(tag):
        try:gpu.check_stop(stop);return action(tag,stop)
        except BaseException:stop.set();raise
    pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='graph20')
    try:
        futures={pool.submit(one,t):t for t in tags}
        for f in as_completed(futures):
            try:f.result()
            except BaseException as exc:errors.append(exc)
    except BaseException:
        stop.set();raise
    finally:pool.shutdown(wait=True)
    if errors:
        raise next((e for e in errors if not isinstance(e,gpu.StopRequested)),errors[0])

def wait_dependency(session,spec,fn,stop=None):
    state=session/'gpu_wait_state.json'
    gpu.release_before_dependency(state,spec['name'])
    beat_at=None
    while True:
        try:gpu.check_stop(stop);return fn()
        except m.Pending as exc:
            beat=beat_at is None or time.monotonic()-beat_at>=600
            gpu.write_state(state,heartbeat=beat,status='WAIT_DEPENDENCY',stage=spec['name'],reason=str(exc))
            if beat:
                print(f'HEARTBEAT {gpu.utc()} WAIT_DEPENDENCY {spec["name"]} {exc}',flush=True);beat_at=time.monotonic()
            gpu.pause(300,stop)
def admit(session,tag,args,stop=None):
    for s in m.SEEDS:
        wait_dependency(session,dict(name='pretrained_'+str(s)),lambda:m.pin_p(tag,s),stop)
    archive=Path(args.fb_zip if tag=='FB' else args.wn_zip).resolve(strict=True)
    m.graph(tag,archive)
    p=m.matrix(tag,args.eta_policy)
    for x in p:x['tag']=tag
    plan=dict(tag=tag,eta_policy=args.eta_policy,archive=str(archive),archive_sha256=m.digest(archive),stages=p)
    atomic(session/'plan.json',plan)
    return plan
def process(session,plan,args,stop=None):
    for spec in plan['stages']:
        gpu.check_stop(stop)
        name=spec['name'];out=session/'stages'/name;out.mkdir();log=session/'stage_logs'/(name+'.log')
        deps={n:m.read(session/'stages'/n/'result.json')['status'] for n in spec['needs']}
        bad={k:v for k,v in deps.items() if v!='COMPLETE'}
        print(f'START_STAGE={session.name}/{name}; LOG={log}',flush=True)
        if bad:c.write(out/'result.json',dict(spec,status='SKIPPED_DEPENDENCY',dependencies=bad))
        elif spec['mode']=='transfer':
            row=wait_dependency(session,spec,lambda:m.transfer(session.name,args.zf10_run),stop)
            c.write(out/'result.json',dict(spec,status='COMPLETE',**row))
        else:
            command=[str(Path(sys.executable).absolute()),'-I','-B','-u',str(ROOT/'worker20.py'),str(session),name]
            def own_invoke(*a,**kw):return invoke(*a,**kw,stop=stop)
            code,timed,pid=gpu.run_stage(command,out,log,session,spec,own_invoke,timeout=args.stage_timeout,lock_path=LOCK,stop=stop)
            m.check(code==0 and not timed,'Technical failure; retained worker log: '+str(log))
        gpu.release_before_dependency(session/'gpu_wait_state.json',name)
        result=m.read(out/'result.json');m.check(result['status'] in ('COMPLETE','RESOURCE_LIMIT','DIVERGED','SKIPPED_DEPENDENCY'),'Unknown terminal state')
        seal(out);print(f'COMMITTED={session.name}/{name}; STATUS={result["status"]}',flush=True)
        m.check(spec['arm'] not in ('P','R','IGPU') or result['status']=='COMPLETE','P/R/IGPU required; stop on missing reference')
def audit_session(session):
    from igpu import audit
    plan=m.read(session/'plan.json');tag=plan['tag'];data=m.graph(tag,plan['archive'])
    views=session/'audit_views';views.mkdir();folders=[];status=[];raw_metrics=[]
    for arm in ('P','R',*m.METHODS):
        for seed in m.SEEDS:
            folder=views/f'{arm}_{seed}';folder.mkdir();folders.append(folder)
            c.write(folder/'run.json',dict(method=arm,seed=seed,dataset=tag,backbone='TransE',evaluation='full'))
            for k in range(4):
                st=session/'stages'/f'evaluate_{arm}_{seed}_p{k}'
                commit=m.read(st/'commit.json');r=m.read(m.verify_file(st/'result.json',commit['files']['result.json']))
                status.append(dict(dataset=tag,method=arm,seed=seed,phase=k,status=r['status']))
                if r['status']!='COMPLETE':continue
                phase=folder/f'p{k}';phase.mkdir()
                for row in m.read(m.verify_file(st/'evaluation.json',commit['files']['evaluation.json']))['metrics']:
                    raw_metrics.append(dict(dataset=tag,method=arm,seed=seed,phase=k,**row))
                for n in ('ranks.u32le','evaluation.json'):
                    source=m.verify_file(st/n,commit['files'][n])
                    os.link(source,phase/n)
            c.write(folder/'COMPLETE.json',dict(files=m.signed_payload(folder)))
    audit.run(folders,data,session/'audit')
    c.csvwrite(session/'audit'/'coverage.csv',status)
    c.csvwrite(session/'audit'/'all_rank_metrics_by_seed.csv',raw_metrics)
    steps=[]
    for spec in plan['stages']:
        if spec['mode']!='evaluate':
            r=m.read(session/'stages'/spec['name']/'result.json')
            steps.append({k:r.get(k) for k in ('name','arm','seed','phase','mode','status','reason','optimizer_steps','seconds','gpu_peak_reserved_mib','eta')})
    c.csvwrite(session/'audit'/'stage_resources.csv',steps)
    c.write(session/'audit'/'zero_fisher_eta.json',m.read(session/'stages/scale_ZF/result.json'))
def run(args):
    import fcntl
    if args.eta_policy=='transfer':m.check(args.zf10_run,'Pass completed --zf10-run for the fixed transfer policy')
    else:m.check(not args.zf10_run,'The grid option does not read a 10% ZF session')
    m.check(sys.platform=='linux' and args.trust_checkpoint,'Linux and --trust-checkpoint required')
    hashes();c.official_check()
    verify_archives(args.fb_zip,args.wn_zip)
    output=(ROOT/'results');output.mkdir(exist_ok=True)
    with (ROOT/'.coordinator.lock').open('a+b') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run=output/('run_'+stamp());run.mkdir()
        for name in ('PREDICTIONS.txt','SCALE_RULE.txt','PACKAGE_SHA256.json'):
            shutil.copyfile(ROOT/name,run/name)
        atomic(run/'admit.json',dict(workers=args.workers,stage_timeout_seconds=args.stage_timeout,fisher_timeout_seconds=list(gpu.FISHER_TIMEOUTS),eta_policy=args.eta_policy,predictions_sha256=m.digest(run/'PREDICTIONS.txt'),scale_rule_sha256=m.digest(run/'SCALE_RULE.txt'),package_sha256_manifest=m.digest(run/'PACKAGE_SHA256.json')))
        atomic(ROOT/'latest_run.json',dict(session=str(run)))
        print('SESSION='+str(run),flush=True)
        for tag in ('FB','WN'):
            session=run/tag;session.mkdir()
            for d in ('stages','stage_logs'):(session/d).mkdir()
            atomic(session/'segment.json',dict(dataset=tag))
            gpu.write_state(session/'gpu_wait_state.json',heartbeat=True,status='QUEUED',stage=None)
        def chain(tag,stop):
            session=run/tag
            try:
                plan=admit(session,tag,args,stop);process(session,plan,args,stop)
                gpu.check_stop(stop)
                gpu.write_state(session/'gpu_wait_state.json',heartbeat=True,status='AUDIT_CPU',stage='audit')
                code,timed,pid=invoke([sys.executable,'-I','-B','-u',str(ROOT/'job20.py'),'audit','--session',str(session)],session/'stage_logs/audit.log',timeout=args.stage_timeout,cpu=True,stop=stop)
                m.check(code==0 and not timed,'CPU audit failed or exceeded stage timeout; preserve residue')
                atomic(session/'complete.json',dict(status='COMPLETE',stages=len(plan['stages']),audit_files=m.signed_payload(session/'audit')))
                gpu.write_state(session/'gpu_wait_state.json',heartbeat=True,status='COMPLETE',stage=None)
            except BaseException as exc:
                atomic(session/'failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
                gpu.write_state(session/'gpu_wait_state.json',heartbeat=True,status='STOPPED_BY_PEER' if isinstance(exc,gpu.StopRequested) else 'STOPPED_ERROR',reason=str(exc))
                raise
        try:
            parallel_chains(('FB','WN'),args.workers,chain)
            atomic(run/'complete.json',dict(status='COMPLETE',files={t:m.digest(run/t/'complete.json') for t in ('FB','WN')}))
            print('COMPLETE=True; run bash report.sh',flush=True)
        except BaseException as exc:
            atomic(run/'failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
            raise
def verify_archives(fb=None,wn=None):
    pins=m.read(ROOT/'DATA_SHA256.json')
    for tag,path in (('FB',fb),('WN',wn)):
        entry=pins[tag];p=Path(path) if path else ROOT.parent/entry['filename']
        m.verify_file(p,entry['sha256']);print('DATA_SHA256_OK '+tag+' '+str(p.resolve()),flush=True)

def remaining_estimate(session):
    plan_path=session/'plan.json'
    if not plan_path.is_file():return dict(known_hours=0,unknown_stages=337,note='Waiting for source admission')
    plan=m.read(plan_path);history={};remaining=[]
    for spec in plan['stages']:
        d=session/'stages'/spec['name'];key=(spec['mode'],spec['arm'])
        if (d/'commit.json').is_file():
            r=m.read(d/'result.json');seconds=r.get('seconds')
            if r['status']=='COMPLETE' and isinstance(seconds,(int,float)) and seconds>0:history.setdefault(key,[]).append(seconds)
        else:remaining.append(spec)
    known=0.;unknown=0
    for spec in remaining:
        times=history.get((spec['mode'],spec['arm']))
        if times:known+=sum(times[-4:])/len(times[-4:])
        else:unknown+=1
    return dict(known_hours=round(known/3600,2),unknown_stages=unknown,remaining_stages=len(remaining),note='Partial estimate only; no shared-GPU waits, retries, CPU audit, or current elapsed deduction')

def inspect(report=False):
    pointer=ROOT/'latest_run.json'
    if not pointer.is_file():print('NOT_STARTED');return
    root=Path(m.read(pointer)['session']);m.check(root.resolve().is_relative_to(ROOT/'results'),'Pointer outside results')
    print('SESSION='+str(root))
    for tag in ('FB','WN'):
        s=root/tag;state=s/'gpu_wait_state.json'
        if not s.is_dir():print(tag,'NOT_STARTED');continue
        names=list((s/'stages').glob('*/commit.json'))
        total=len(m.read(s/'plan.json')['stages']) if (s/'plan.json').is_file() else 337
        print(tag,f'SEALED={len(names)}/{total}',json.dumps(m.read(state) if state.exists() else {},ensure_ascii=False))
        print(tag,'REMAINING_TIME='+json.dumps(remaining_estimate(s),ensure_ascii=False))
    if (root/'failure.json').exists():print('STOPPED_ERROR',m.read(root/'failure.json')['message'])
    if not report or not (root/'complete.json').is_file():return
    out=root/('report_'+stamp()+'.zip')
    main=m.read(root/'complete.json')
    with zipfile.ZipFile(out,'x',zipfile.ZIP_DEFLATED) as z:
        for tag,h in main['files'].items():
            sub=root/tag
            for n,d in m.read(m.verify_file(sub/'complete.json',h))['audit_files'].items():
                if Path(n).suffix in ('.csv','.txt','.json'):z.write(m.verify_file(sub/'audit'/n,d),tag+'/'+n)
        for n in ('PREDICTIONS.txt','SCALE_RULE.txt','admit.json','complete.json'):z.write(root/n,n)
    print('REPORT_ZIP='+str(out));print('REPORT_SHA256='+m.digest(out))
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('run','progress','report','verify','verify-data','audit','plan'))
    p.add_argument('--trust-checkpoint',action='store_true');p.add_argument('--fb-zip');p.add_argument('--wn-zip')
    p.add_argument('--eta-policy',choices=('transfer','grid'),default='grid');p.add_argument('--zf10-run')
    p.add_argument('--workers',type=int,choices=(1,2),default=2);p.add_argument('--stage-timeout',type=int,default=7200);p.add_argument('--session');a=p.parse_args()
    m.check(a.stage_timeout>0,'Positive stage timeout required')
    if a.mode=='verify':hashes();print('PACKAGE_SHA256_OK')
    elif a.mode=='verify-data':verify_archives(a.fb_zip,a.wn_zip)
    elif a.mode=='audit':
        session=Path(a.session).resolve(strict=True);m.check(session.is_relative_to(ROOT/'results'),'Audit outside new package results');audit_session(session)
    elif a.mode=='run':
        p.error('Pass both official ZIPs and one --eta-policy') if not all((a.fb_zip,a.wn_zip,a.eta_policy)) else run(a)
    elif a.mode=='plan':
        print(json.dumps({t:m.matrix(t,a.eta_policy) for t in m.SPECS},indent=2))
    else:inspect(a.mode=='report')
if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *unused:(_ for _ in ()).throw(SystemExit('Coordinator stopped')))
    main()
