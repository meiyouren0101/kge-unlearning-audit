"""Private single-worker Schema job, immutable outputs and dependency-aware audit."""
import argparse,os,sys,json,shutil,time,traceback,zipfile,signal
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from igpu import bootstrap
import common as c,gpu_wait as gpu,matrix20 as m,schema_inputs as inp
from job20 import stamp,atomic,seal,invoke,hashes,LOCK
ROOT=m.ROOT

def env_check(production=False):
    import sklearn,torch,numpy,scipy
    from worker_schema import sklearn_contract
    d=dict(python=sys.version,torch=torch.__version__,numpy=numpy.__version__,scipy=scipy.__version__,**sklearn_contract())
    if production:m.check(torch.__version__.split('+')[0]=='2.1.1' and torch.version.cuda=='12.1','Production requires torch2.1.1+cu121')
    c.official_check();print('ENVIRONMENT='+json.dumps(d,ensure_ascii=False));return d

def process(session):
    plan=m.read(session/'plan.json')
    for spec in plan['stages']:
        name=spec['name'];out=session/'stages'/name
        if (out/'commit.json').is_file():
            for n,h in m.read(out/'commit.json')['files'].items():m.verify_file(out/n,h)
            print('REUSED='+session.name+'/'+name,flush=True);continue
        m.check(not out.exists(),'Unsealed residue exists; stop and preserve '+str(out));out.mkdir()
        deps={n:m.read(session/'stages'/n/'result.json')['status'] for n in spec['needs']}
        bad={n:s for n,s in deps.items() if s!='COMPLETE'}
        if bad:c.write(out/'result.json',dict(spec,status='SKIPPED_DEPENDENCY',dependencies=bad))
        else:
            log=session/'stage_logs'/(name+'.log');print('START_STAGE='+session.name+'/'+name+'; LOG='+str(log),flush=True)
            code,timed,pid=gpu.run_stage([sys.executable,'-I','-B','-u',str(ROOT/'worker_schema.py'),str(session),name],
                out,log,session,spec,invoke,timeout=7200,lock_path=LOCK)
            m.check(code==0 and not timed,'Worker error; preserve '+str(log))
        seal(out);print('COMMITTED='+session.name+'/'+name+'; STATUS='+m.read(out/'result.json')['status'],flush=True)
    gpu.release_before_dependency(session/'gpu_wait_state.json','audit')
    atomic(session/'gpu_complete.json',dict(status='COMPLETE',stages=len(plan['stages'])))

def pending_audit(session,a20):
    return list(dict.fromkeys(p for k in range(4) for p in inp.waiting(session.name,k,a20)))

def run_audits(root,a20):
    todo=[root/k for k in inp.ROSTERS if not (root/k/'complete.json').is_file()]
    last={}
    while todo:
        for session in todo[:]:
            gpu.release_before_dependency(session/'gpu_wait_state.json','audit')
            missing=pending_audit(session,a20)
            if missing:
                beat=session not in last or time.monotonic()-last[session]>=600
                gpu.write_state(session/'gpu_wait_state.json',heartbeat=beat,status='WAIT_DEPENDENCY',stage='audit',missing_seals=missing)
                if beat:print('HEARTBEAT '+gpu.utc()+' WAIT_DEPENDENCY '+session.name+' '+str(missing[0]),flush=True);last[session]=time.monotonic()
                continue
            gpu.write_state(session/'gpu_wait_state.json',heartbeat=True,status='AUDIT_CPU',stage='audit')
            code,timed,_=invoke([sys.executable,'-I','-B','-u',str(ROOT/'job_schema.py'),'audit','--session',str(session),'--a20-session',str(a20)],
                session/'stage_logs/audit.log',timeout=7200,cpu=True)
            m.check(code==0 and not timed,'Audit failed; preserve '+str(session))
            gpu.write_state(session/'gpu_wait_state.json',heartbeat=True,status=m.read(session/'complete.json')['status'],stage=None)
            todo.remove(session)
        if todo:gpu.pause(300)

def run(args):
    import fcntl
    m.check(sys.platform=='linux' and args.trust_checkpoint,'Linux and explicit --trust-checkpoint required')
    hashes();environment=env_check(True)
    archives={k:Path(getattr(args,k.lower()+'_zip')).absolute() for k in inp.ROSTERS}
    for k,p in archives.items():inp.graph(k,p)
    for tag in ('FB','WN'):
        for seed in m.SEEDS:m.pin_p(tag,seed)
    # Only C's lock. A/B/ZeroFisher/YAGO PIDs are neither searched nor stopped.
    with (ROOT/'.coordinator.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out=ROOT/'results';out.mkdir(exist_ok=True);root=out/('run_'+stamp());root.mkdir()
        for n in ('PREDICTIONS.txt','protocol.json','PACKAGE_SHA256.json'):shutil.copyfile(ROOT/n,root/n)
        atomic(root/'admit.json',dict(a20_session=str(Path(args.a20_session).absolute()),environment=environment,
            predictions_sha256=m.digest(root/'PREDICTIONS.txt'),protocol_sha256=m.digest(root/'protocol.json'),
            package_manifest_sha256=m.digest(root/'PACKAGE_SHA256.json')))
        atomic(ROOT/'latest_run.json',dict(session=str(root)));print('SESSION='+str(root),flush=True)
        for k in inp.ROSTERS:
            session=root/k;session.mkdir();(session/'stages').mkdir();(session/'stage_logs').mkdir()
            atomic(session/'plan.json',dict(key=k,archive=str(archives[k]),archive_sha256=m.digest(archives[k]),stages=inp.stage_plan(k)))
            gpu.write_state(session/'gpu_wait_state.json',heartbeat=True,status='QUEUED',stage=None)
        try:
            for k in inp.ROSTERS:process(root/k)
            run_audits(root,args.a20_session)
            all_ok=all(m.read(root/k/'complete.json')['status']=='COMPLETE' for k in inp.ROSTERS)
            atomic(root/'complete.json',dict(status='COMPLETE' if all_ok else 'COMPLETE_WITH_UNAVAILABLE',files={k:m.digest(root/k/'complete.json') for k in inp.ROSTERS}))
            print('ALL_TERMINAL=True; read report.sh for coverage and unavailable stages.',flush=True)
        except BaseException as exc:
            atomic(root/'failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
            for k in inp.ROSTERS:
                if not (root/k/'complete.json').exists():
                    gpu.write_state(root/k/'gpu_wait_state.json',heartbeat=True,status='STOPPED_ERROR',reason=str(exc))
            raise

def inspect(report=False):
    pointer=ROOT/'latest_run.json'
    if not pointer.exists():print('NOT_STARTED');return
    root=Path(m.read(pointer)['session']).resolve();m.check(root.is_relative_to((ROOT/'results').resolve()),'Unsafe result pointer')
    print('SESSION='+str(root));lines=['SESSION='+str(root)]
    terminal=m.read(root/'complete.json')['status'] if (root/'complete.json').is_file() else 'INCOMPLETE'
    print('OVERALL_STATUS='+terminal);lines.append('OVERALL_STATUS='+terminal)
    if (root/'failure.json').exists():print('STOPPED_ERROR='+m.read(root/'failure.json')['message'])
    for key in inp.ROSTERS:
        s=root/key;stages=list((s/'stages').glob('*/commit.json'));states={}
        for sealpath in stages:
            sealrow=m.read(sealpath);r=m.read(m.verify_file(sealpath.with_name('result.json'),sealrow['files']['result.json']))
            states[r['status']]=states.get(r['status'],0)+1
        state=m.read(s/'gpu_wait_state.json') if (s/'gpu_wait_state.json').exists() else {}
        total=len(inp.stage_plan(key));text=f'{key} SEALED={len(stages)}/{total}; STATES={states}; AUDIT_COMPLETE={(s/"complete.json").is_file()}'
        print(text);print(json.dumps(state,ensure_ascii=False));lines.append(text);lines.append(json.dumps(state,ensure_ascii=False))
    if not report:return
    zpath=root/('report_'+stamp()+'.zip')
    # Small only: explicit allowlist of own outputs. No upstream cache/checkpoint.
    with zipfile.ZipFile(zpath,'x',zipfile.ZIP_DEFLATED) as z:
        z.writestr('report.txt','\n'.join(lines)+'\n')
        for n in ('admit.json','PREDICTIONS.txt','protocol.json','complete.json','failure.json'):
            if (root/n).is_file():z.write(root/n,n)
        for key in inp.ROSTERS:
            s=root/key
            for f in (s/'stages').glob('*/result.json'):z.write(f,f.relative_to(root).as_posix())
            if (s/'complete.json').exists():
                for n,h in m.read(s/'complete.json')['audit_files'].items():
                    if Path(n).suffix in ('.csv','.json','.txt'):z.write(m.verify_file(s/'audit'/n,h),key+'/audit/'+n)
            for f in (s/'audit_views').glob('*/*/source_receipt.json'):z.write(f,f.relative_to(root).as_posix())
    print('REPORT_ZIP='+str(zpath));print('REPORT_SHA256='+m.digest(zpath))

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('all','verify','check-env','check-data','progress','report','audit','plan'))
    p.add_argument('--trust-checkpoint',action='store_true');p.add_argument('--a20-session');p.add_argument('--session')
    for k in inp.ROSTERS:p.add_argument('--'+k.lower()+'-zip')
    a=p.parse_args()
    if a.mode=='all':
        if not a.a20_session or not all(getattr(a,k.lower()+'_zip') for k in inp.ROSTERS):p.error('Supply four ZIP paths and exact --a20-session')
        run(a)
    elif a.mode=='verify':hashes();print('PACKAGE_SHA256_OK')
    elif a.mode=='check-env':env_check(False)
    elif a.mode=='check-data':
        for k in inp.ROSTERS:
            v=getattr(a,k.lower()+'_zip');m.check(v,'Missing archive '+k);d=inp.graph(k,v);print(k,d['counts'])
    elif a.mode=='audit':
        from audit_schema import build
        s=Path(a.session).resolve();m.check(s.is_relative_to((ROOT/'results').resolve()),'Audit outside own results');build(s,a.a20_session)
    elif a.mode=='plan':print(json.dumps({k:inp.stage_plan(k) for k in inp.ROSTERS},indent=2))
    else:inspect(a.mode=='report')
if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *unused:(_ for _ in ()).throw(SystemExit('Own coordinator stopped')))
    main()
