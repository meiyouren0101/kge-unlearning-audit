"""CPU scheduler fixtures; no GPU, production job, or external process touched."""
import json, sys, tempfile, threading, time, subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch
import job20 as j, gpu_wait as g, matrix20 as m

def snapshot(own=False,foreign=False,growth=0):
    return dict(total_mib=24564,free_mib=23000-(500 if own else 0)-growth,
                compute=([dict(pid=10,used_mib=500)] if own else [])+([dict(pid=99,used_mib=200)] if foreign else []))

class Scheduling(TestCase):
    def test_chains_really_overlap(self):
        barrier=threading.Barrier(2);seen=[]
        def action(tag,stop):barrier.wait(timeout=5);seen.append(tag)
        j.parallel_chains(('FB','WN'),2,action)
        self.assertEqual(set(seen),{'FB','WN'})

    def test_one_worker_stays_sequential(self):
        seen=[]
        j.parallel_chains(('FB','WN'),1,lambda t,s:seen.append(t))
        self.assertEqual(seen,['FB','WN'])

    def test_peer_failure_sets_cancel_and_preserves_original_error(self):
        ready=threading.Event();cancelled=[]
        def action(tag,stop):
            if tag=='FB':ready.wait(5);raise ValueError('original technical error')
            ready.set()
            try:g.pause(30,stop)
            except g.StopRequested:cancelled.append(True);raise
        with self.assertRaisesRegex(ValueError,'original technical error'):j.parallel_chains(('FB','WN'),2,action)
        self.assertEqual(cancelled,[True])

    def test_real_subprocess_timeout_stops_only_owned_child(self):
        with tempfile.TemporaryDirectory() as t:
            result=j.invoke([sys.executable,'-c','import time;time.sleep(20)'],Path(t)/'log',timeout=.15,cpu=True,poll_seconds=.02)
            self.assertTrue(result[1]);self.assertNotEqual(result[0],0)

    def test_real_subprocess_peer_cancel(self):
        stop=threading.Event();timer=threading.Timer(.2,stop.set);timer.start()
        try:
            with tempfile.TemporaryDirectory() as t:
                with self.assertRaises(g.StopRequested):j.invoke([sys.executable,'-c','import time;time.sleep(20)'],Path(t)/'log',cpu=True,stop=stop,poll_seconds=.02)
        finally:timer.cancel()

    def test_own_oom_all_baselines_both_graph_names(self):
        for tag in ('FB','WN'):
            for arm in m.METHODS:
                w=g.Watcher(snapshot());w.observe(snapshot(own=True),10)
                self.assertEqual(g.own_baseline_oom(dict(arm=arm),w),arm!='IGPU',(tag,arm))

    def test_protected_not_own_baseline(self):
        w=g.Watcher(snapshot());w.observe(snapshot(own=True),10)
        for arm in ('P','R','IGPU'):self.assertFalse(g.own_baseline_oom(dict(arm=arm),w))

    def test_competition_not_forgotten_when_last16_roll_over(self):
        w=g.Watcher(snapshot());w.observe(snapshot(own=True,foreign=True),10)
        for _ in range(50):w.observe(snapshot(own=True),10)
        self.assertFalse(g.own_baseline_oom(dict(arm='Fisher'),w))

    def test_start_competitor_and_external_growth_count(self):
        for w,s in ((g.Watcher(snapshot(foreign=True)),snapshot(own=True)),(g.Watcher(snapshot()),snapshot(own=True,growth=300))):
            w.observe(s,10);self.assertFalse(g.own_baseline_oom(dict(arm='RL'),w))

    def test_no_own_telemetry_is_not_own_baseline_proof(self):
        self.assertFalse(g.own_baseline_oom(dict(arm='RL'),g.Watcher(snapshot())))

    def test_timeout_limits(self):
        self.assertEqual([g.stage_timeout(dict(mode='delete',arm='Fisher'),i) for i in range(3)],[21600,43200,86400])
        for arm in (*m.METHODS,'R','P'):
            self.assertEqual(g.stage_timeout(dict(mode='evaluate',arm=arm)),7200)
        with self.assertRaises(ValueError):g.stage_timeout(dict(mode='delete',arm='Fisher'),3)

    def simulate(self,root,arm,mode,outcomes):
        session=root/'FB';folder=session/'stages'/'item';folder.mkdir(parents=True)
        log=session/'worker.log';spec=dict(name='item',arm=arm,mode=mode,estimated_peak_mib=1000,seed=0,phase=0,needs=[])
        calls=[];leases=[]
        def start(*a,**kw):
            lease=Mock();leases.append(lease);return dict(snapshot(foreign=True),_lease=lease)
        def invoke(cmd,log,timeout,**kw):
            calls.append(timeout);kw['monitor'].started(10);Path(log).write_text('fixture')
            outcome=outcomes[len(calls)-1]
            if outcome=='oom':j.c.write(folder/'gpu_oom.json',dict(stage='item',status='GPU_OOM',snapshot=snapshot(own=True,foreign=True),torch_peak_reserved_mib=500))
            elif outcome=='own':j.c.write(folder/'gpu_oom.json',dict(stage='item',status='GPU_OOM',snapshot=snapshot(own=True),torch_peak_reserved_mib=500));kw['monitor'].competition_seen=False
            elif outcome=='ok':
                j.c.write(folder/'result.json',dict(status='COMPLETE',seconds=1))
                j.c.write(folder/'determinism_bridge.json',dict(status='PASS',retry_attempts=len(calls)-1))
            return 0,outcome=='timeout',10
        def cooldown(seconds,stop):
            self.assertEqual(seconds,60);leases[-1].release.assert_called_once()
        with patch.object(g,'wait_for_gpu',side_effect=start),patch.object(g,'pause',side_effect=cooldown):
            result=g.run_stage([],folder,log,session,spec,invoke)
        return calls,folder,result

    def test_fisher_timeout_three_attempts_then_resource(self):
        with tempfile.TemporaryDirectory() as t:
            calls,folder,result=self.simulate(Path(t),'Fisher','delete',['timeout']*3)
            self.assertEqual(calls,[21600,43200,86400]);self.assertEqual(m.read(folder/'result.json')['timeout_count'],3)
            self.assertEqual(m.read(folder/'result.json')['status'],'RESOURCE_LIMIT');self.assertFalse(result[1])

    def test_oom_does_not_spend_fisher_timeout_budget(self):
        with tempfile.TemporaryDirectory() as t:
            calls,folder,result=self.simulate(Path(t),'Fisher','delete',['oom','timeout','ok'])
            self.assertEqual(calls,[21600,21600,43200]);self.assertEqual(m.read(folder/'result.json')['status'],'COMPLETE')

    def test_four_competed_ooms_resource_limit(self):
        with tempfile.TemporaryDirectory() as t:
            calls,folder,_=self.simulate(Path(t),'RL','delete',['oom']*4)
            self.assertEqual(len(calls),4);self.assertEqual(m.read(folder/'result.json')['oom_count'],4)

    def test_own_oom_no_retry_in_FB_session(self):
        with tempfile.TemporaryDirectory() as t:
            calls,folder,_=self.simulate(Path(t),'Fisher','delete',['own'])
            self.assertEqual(len(calls),1);self.assertFalse(m.read(folder/'result.json')['automatic_retry'])

    def test_generic_timeout_no_retry(self):
        with tempfile.TemporaryDirectory() as t:
            calls,folder,_=self.simulate(Path(t),'RL','delete',['timeout'])
            self.assertEqual(calls,[7200]);self.assertEqual(m.read(folder/'result.json')['status'],'RESOURCE_LIMIT')

    def test_protected_timeouts_stop_without_terminal_resource(self):
        for arm in ('P','R','IGPU'):
            with tempfile.TemporaryDirectory() as t:
                with self.assertRaises(g.ResourceLimit):self.simulate(Path(t),arm,'evaluate',['timeout'])
                self.assertFalse((Path(t)/'FB/stages/item/result.json').exists())
                self.assertTrue((Path(t)/'FB/stages/item/resource_failure.json').exists())

    def test_resource_dependent_skipped_but_independent_runs(self):
        with tempfile.TemporaryDirectory() as t:
            session=Path(t)/'FB';(session/'stages').mkdir(parents=True);(session/'stage_logs').mkdir()
            rows=[dict(name=n,mode='delete',arm='RL',needs=needs) for n,needs in [('bad',[]),('child',['bad']),('independent',[])]]
            invoked=[]
            def run(cmd,out,log,session,spec,*args,**kw):
                invoked.append(spec['name']);j.c.write(out/'result.json',dict(status='RESOURCE_LIMIT' if spec['name']=='bad' else 'COMPLETE'));return 0,False,10
            with patch.object(g,'run_stage',side_effect=run),patch.object(g,'release_before_dependency'):
                j.process(session,dict(stages=rows),SimpleNamespace(stage_timeout=7200))
            self.assertEqual(invoked,['bad','independent']);self.assertEqual(m.read(session/'stages/child/result.json')['status'],'SKIPPED_DEPENDENCY')

    def test_eta_estimate_never_calls_partial_time_total(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);a=p/'stages/a';a.mkdir(parents=True)
            rows=[dict(name=n,mode='delete',arm=arm) for n,arm in [('a','RL'),('b','RL'),('c','Fisher')]]
            j.c.write(p/'plan.json',dict(stages=rows));j.c.write(a/'commit.json',{});j.c.write(a/'result.json',dict(status='COMPLETE',seconds=3600))
            row=j.remaining_estimate(p);self.assertEqual(row['known_hours'],1);self.assertEqual(row['unknown_stages'],1)

    def test_default_grid_and_workers6(self):
        with patch.object(sys,'argv',['job20.py','run','--fb-zip','f','--wn-zip','w']),patch.object(j,'run') as run:
            j.main();args=run.call_args.args[0];self.assertEqual((args.eta_policy,args.workers,args.stage_timeout),('grid',6,7200));self.assertIsNone(args.zf10_run)

    def test_archive_hash_check_rejects_wrong_bytes(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'bad.zip';p.write_bytes(b'wrong')
            with self.assertRaises(ValueError):j.verify_archives(str(p),str(p))

    def test_dependency_wait_cancel_is_interruptible(self):
        with tempfile.TemporaryDirectory() as t:
            stop=threading.Event();released=[]
            def retry(seconds,event):self.assertEqual(seconds,300);stop.set();g.check_stop(stop)
            with patch.object(g,'release_before_dependency',side_effect=lambda *a:released.append(True)),patch.object(g,'pause',side_effect=retry):
                with self.assertRaises(g.StopRequested):j.wait_dependency(Path(t),dict(name='P'),lambda:(_ for _ in ()).throw(m.Pending('missing seal')),stop)
            self.assertEqual(released,[True])

    def test_progress_shows_both_chain_states_and_remaining(self):
        import contextlib,io
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);run=root/'results/run_fixture';run.mkdir(parents=True)
            j.c.write(root/'latest_run.json',dict(session=str(run)))
            for tag,status in [('FB','WAIT_GPU'),('WN','RUNNING')]:
                s=run/tag;(s/'stages').mkdir(parents=True);j.c.write(s/'gpu_wait_state.json',dict(status=status,stage='item'))
            out=io.StringIO()
            with patch.object(j,'ROOT',root),contextlib.redirect_stdout(out):j.inspect()
            text=out.getvalue()
            for expected in ('FB SEALED=0/337','WN SEALED=0/337','WAIT_GPU','RUNNING','FB/seed0 REMAINING_TIME','WN/seed123 REMAINING_TIME'):self.assertIn(expected,text)
