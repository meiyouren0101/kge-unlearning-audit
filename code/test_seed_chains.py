"""Six-chain scheduler fixtures. No server/GPU connections."""
import contextlib, io, json, sys, tempfile, threading, time
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch
import job20 as j, gpu_wait as g, matrix20 as m

def committed(session,name,status='COMPLETE'):
    out=session/'stages'/name;out.mkdir(parents=True,exist_ok=True)
    j.c.write(out/'result.json',dict(status=status,seconds=60))
    j.seal(out)

class SeedChains(TestCase):
    def test_exact_partition_674_stages_no_added_or_missing(self):
        for policy in ('grid','transfer'):
            for tag in ('FB','WN'):
                plan=dict(stages=m.matrix(tag,policy))
                partitions=[j.chain_stages(plan,s) for s in m.SEEDS]
                self.assertEqual(list(map(len,partitions)),[113,112,112])
                names=[r['name'] for rs in partitions for r in rs]
                self.assertEqual(len(names),len(set(names)))
                self.assertEqual(set(names),{r['name'] for r in plan['stages']})

    def test_requests_monotone_dependencies_topological(self):
        for tag in ('FB','WN'):
            plan=dict(stages=m.matrix(tag,'grid'))
            for seed in m.SEEDS:
                rows=j.chain_stages(plan,seed);seen=set()
                self.assertEqual([r['phase'] for r in rows],sorted(r['phase'] for r in rows))
                for row in rows:
                    for dep in row['needs']:
                        self.assertTrue(dep in seen or seed!=0 and dep=='scale_ZF',(row,dep))
                    seen.add(row['name'])

    def test_only_seed0_owns_grid_and_producer_queued_first(self):
        self.assertEqual(j.chain_ids()[:2],[('FB',0),('WN',0)])
        for seed in m.SEEDS:
            grid=[r for r in j.chain_stages(dict(stages=m.matrix('FB','grid')),seed) if r['name']=='scale_ZF']
            self.assertEqual(len(grid),int(seed==0))

    def test_all_six_actions_overlap(self):
        barrier=threading.Barrier(6);seen=[]
        def action(identity,stop):barrier.wait(10);seen.append(identity)
        j.parallel_chains(j.chain_ids(),6,action)
        self.assertEqual(set(seen),set(j.chain_ids()))

    def test_workers2_concurrency_bound_and_workers1_producer_order(self):
        active=[0];maximum=[0];lock=threading.Lock()
        def action(identity,stop):
            with lock:active[0]+=1;maximum[0]=max(maximum[0],active[0])
            stop.wait(.03)
            with lock:active[0]-=1
        j.parallel_chains(j.chain_ids(),2,action);self.assertEqual(maximum[0],2)
        seen=[];j.parallel_chains(j.chain_ids(),1,lambda t,s:seen.append(t))
        self.assertEqual(seen,j.chain_ids())

    def test_failure_cancels_other_five(self):
        barrier=threading.Barrier(6);cancelled=[]
        def action(identity,stop):
            barrier.wait(10)
            if identity==('FB',0):raise ValueError('primary chain technical failure')
            try:g.pause(20,stop)
            except g.StopRequested:cancelled.append(identity);raise
        with self.assertRaisesRegex(ValueError,'primary chain'):j.parallel_chains(j.chain_ids(),6,action)
        self.assertEqual(len(cancelled),5)

    def test_dependency_needs_commit_not_just_result(self):
        with tempfile.TemporaryDirectory() as t:
            s=Path(t);d=s/'stages/scale_ZF';d.mkdir(parents=True);j.c.write(d/'result.json',dict(status='COMPLETE'))
            with self.assertRaises(m.Pending):j.dependency_status(s,'scale_ZF')
            j.seal(d);self.assertEqual(j.dependency_status(s,'scale_ZF'),'COMPLETE')
            (d/'result.json').write_text(json.dumps(dict(status='RESOURCE_LIMIT')))
            with self.assertRaises(ValueError):j.dependency_status(s,'scale_ZF')

    def test_cross_seed_grid_wait_then_same_sealed_eta(self):
        with tempfile.TemporaryDirectory() as t:
            s=Path(t)/'FB';(s/'stages').mkdir(parents=True);cf=j.chain_folder(s,42);cf.mkdir(parents=True)
            attempts=[]
            def wait(seconds,stop):
                self.assertEqual(seconds,300);attempts.append(1);committed(s,'scale_ZF')
            with patch.object(g,'release_before_dependency') as release,patch.object(g,'pause',side_effect=wait):
                status=j.wait_dependency(cf,dict(name='train_ZF_42_p0'),lambda:j.dependency_status(s,'scale_ZF'))
            self.assertEqual(status,'COMPLETE');self.assertEqual(len(attempts),1);release.assert_called_once()
            self.assertFalse((s/'gpu_wait_state.json').exists());self.assertTrue((cf/'gpu_wait_state.json').exists())

    def test_per_chain_status_and_peak_paths_distinct(self):
        with tempfile.TemporaryDirectory() as t:
            s=Path(t)/'FB';(s/'stages').mkdir(parents=True);(s/'stage_logs').mkdir()
            targets=[]
            def run(cmd,out,log,session,spec,*a,**kw):
                targets.append((kw['state_path'],kw['peaks_path']));j.c.write(out/'result.json',dict(status='COMPLETE'));return 0,False,10
            for seed in m.SEEDS:
                cf=j.chain_folder(s,seed);cf.mkdir(parents=True)
                spec=dict(name='toy_'+str(seed),arm='RL',mode='delete',needs=[])
                with patch.object(g,'run_stage',side_effect=run),patch.object(g,'release_before_dependency'):
                    j.process(s,dict(stages=[spec]),SimpleNamespace(stage_timeout=7200),rows=[spec],state_root=cf)
            self.assertEqual(len({a for a,b in targets}),3);self.assertEqual(len({b for a,b in targets}),3)
            self.assertEqual(len(list((s/'stages').glob('*/commit.json'))),3)

    def test_audit_barrier_refuses_missing_seed(self):
        with tempfile.TemporaryDirectory() as t,patch.object(j,'invoke') as invoke:
            root=Path(t);plans={tag:dict(stages=[]) for tag in ('FB','WN')}
            for tag,seed in j.chain_ids()[:-1]:
                cf=j.chain_folder(root/tag,seed);cf.mkdir(parents=True);j.c.write(cf/'complete.json',dict(status='COMPLETE'))
            with self.assertRaises(FileNotFoundError):j.finish_audits(root,plans,SimpleNamespace(stage_timeout=7200))
            invoke.assert_not_called()

    def test_audit_barrier_runs_once_per_graph_after_six(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);plans={tag:dict(stages=[]) for tag in ('FB','WN')}
            for tag,seed in j.chain_ids():
                cf=j.chain_folder(root/tag,seed);cf.mkdir(parents=True);j.c.write(cf/'complete.json',dict(status='COMPLETE'))
            order=[]
            def invoke(cmd,*a,**kw):
                session=Path(cmd[-1]);order.append(session.name);self.assertTrue(kw['cpu']);(session/'audit').mkdir();j.c.write(session/'audit/example.json',{});return 0,False,10
            with patch.object(j,'invoke',side_effect=invoke):j.finish_audits(root,plans,SimpleNamespace(stage_timeout=7200))
            self.assertEqual(order,['FB','WN'])

    def test_remaining_estimate_uses_only_selected_seed(self):
        with tempfile.TemporaryDirectory() as t:
            s=Path(t);rows=[dict(name=n,arm='RL',mode='delete',seed=seed,phase=k) for n,seed,k in [('a',0,0),('b',0,1),('c',42,0),('d',42,1)]]
            j.c.write(s/'plan.json',dict(stages=rows));committed(s,'a')
            a=j.remaining_estimate(s,0);b=j.remaining_estimate(s,42)
            self.assertEqual((a['remaining_stages'],a['unknown_stages']),(1,0));self.assertEqual((b['remaining_stages'],b['unknown_stages']),(2,2))

    def test_cli_workers1_through6_and_default6(self):
        for n in range(1,7):
            with patch.object(sys,'argv',['job20.py','run','--workers',str(n),'--fb-zip','f','--wn-zip','w']),patch.object(j,'run') as run:
                j.main();self.assertEqual(run.call_args.args[0].workers,n)
        for n in ('0','7'):
            with patch.object(sys,'argv',['job20.py','plan','--workers',n]),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):j.main()

    def test_cancel_marks_queued_but_preserves_terminal(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for tag,seed in j.chain_ids():g.write_state(j.chain_folder(root/tag,seed)/'gpu_wait_state.json',status='QUEUED')
            p=j.chain_folder(root/'FB',0)/'gpu_wait_state.json';g.write_state(p,status='COMPLETE')
            j.mark_cancelled(root,'test failure')
            self.assertEqual(m.read(p)['status'],'COMPLETE')
            for tag,seed in j.chain_ids()[1:]:self.assertEqual(m.read(j.chain_folder(root/tag,seed)/'gpu_wait_state.json')['status'],'STOPPED_BY_PEER')

    def test_45s_lease_and_wait_rules_unchanged(self):
        self.assertEqual((g.START_GRACE_SECONDS,g.POLL_SECONDS,g.HEARTBEAT_SECONDS),(45,60,600))
        self.assertEqual(g.SHARED_LOCK,'/mnt/new4T/lsb/.graphdpo_gpu0_admission.lock')
        timer=Mock();module=SimpleNamespace(Timer=Mock(return_value=timer),Lock=threading.Lock)
        with patch.object(g,'threading',module):
            lease=g.StartLease('unused');lease.file=Mock();lease.started()
            self.assertEqual(module.Timer.call_args.args[0],45);timer.start.assert_called_once()
            # This synthetic lease never owns an OS lock.
            lease.file=None;g.pending_leases().clear()

    def test_scientific_files_and_embedded_audit_byte_proof(self):
        import hashlib,ast
        proof=m.read(m.ROOT/'validation/V3_SCIENCE_COMPARE.json')
        for row in proof['files']:
            self.assertTrue(row['byte_identical']);self.assertEqual(m.digest(m.ROOT/row['path']),row['v2_sha256'])
        source=(m.ROOT/'job20.py').read_bytes();tree=ast.parse(source)
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='audit_session')
        data=b''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
        self.assertEqual(hashlib.sha256(data).hexdigest(),proof['embedded_audit_session']['v2_sha256'])

    def test_full_674_stage_synthetic_coordinator_six_chains(self):
        # Actual coordinator/thread/filesystem flow; GPU science is a tiny terminal stub.
        import safe_stop_v2
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);fb=root/'fb.zip';wn=root/'wn.zip';fb.write_bytes(b'fixture');wn.write_bytes(b'fixture')
            for n in ('PREDICTIONS.txt','SCALE_RULE.txt','PACKAGE_SHA256.json'):(root/n).write_text('fixture')
            args=SimpleNamespace(eta_policy='grid',zf10_run=None,trust_checkpoint=True,fb_zip=str(fb),wn_zip=str(wn),workers=6,stage_timeout=7200)
            counts=[];audits=[];lock=threading.Lock()
            def fake_worker(cmd,out,log,session,spec,*a,**kw):
                with lock:counts.append((session.name,spec['name']))
                j.c.write(out/'result.json',dict(status='COMPLETE',seconds=1))
                return 0,False,10
            def fake_audit(cmd,log,**kw):
                self.assertEqual(len(counts),674);self.assertTrue(kw['cpu'])
                session=Path(cmd[-1]);(session/'audit').mkdir();j.c.write(session/'audit/stub.json',{});audits.append(session.name)
                return 0,False,10
            fake_fcntl=SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=lambda *a:None)
            real_pause=g.pause
            def fast_dependency(seconds,stop=None):
                return real_pause(.005 if seconds==300 else seconds,stop)
            with patch.object(j,'ROOT',root),patch.object(j,'sys',SimpleNamespace(platform='linux',executable=sys.executable)),patch.dict(sys.modules,{'fcntl':fake_fcntl}),patch.object(j,'hashes'),patch.object(j.c,'official_check'),patch.object(j,'verify_archives'),patch.object(m,'graph'),patch.object(m,'pin_p'),patch.object(safe_stop_v2,'matching_processes',return_value=[]),patch.object(g,'run_stage',side_effect=fake_worker),patch.object(j,'invoke',side_effect=fake_audit),patch.object(g,'pause',side_effect=fast_dependency),contextlib.redirect_stdout(io.StringIO()):
                j.run(args)
            run=Path(m.read(root/'latest_run.json')['session'])
            self.assertEqual(len(counts),len(set(counts)));self.assertEqual(audits,['FB','WN'])
            self.assertEqual(m.read(run/'complete.json')['status'],'COMPLETE')
            for tag,seed in j.chain_ids():
                cf=j.chain_folder(run/tag,seed)
                self.assertEqual(m.read(cf/'gpu_wait_state.json')['status'],'COMPLETE')
                self.assertEqual(m.read(cf/'complete.json')['stages'],113 if seed==0 else 112)
