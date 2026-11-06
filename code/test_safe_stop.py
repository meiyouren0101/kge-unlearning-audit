import copy, tempfile, os, contextlib
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import safe_stop_v2 as s

def fixture():
    return dict(committed={'FB':0,'WN':0},states={t:dict(status='WAIT_GPU') for t in ('FB','WN')},
        processes=[dict(pid=101,uid=123,script='job20.py',cmd=['python','job20.py','run'],start_ticks=500,cwd='/tmp/v2')])

class SafeStop(TestCase):
    def test_waiting_zero_stage_coordinator_only(self):self.assertEqual(s.validate_waiting(fixture(),123,[])['pid'],101)
    def test_completed_stage_refused(self):
        f=fixture();f['committed']['FB']=1
        with self.assertRaisesRegex(ValueError,'sealed'):s.validate_waiting(f,123,[])
    def test_running_worker_refused(self):
        f=fixture();f['processes'].append(dict(pid=102,uid=123,script='worker20.py'))
        with self.assertRaisesRegex(ValueError,'worker'):s.validate_waiting(f,123,[])
    def test_live_descendant_refused(self):
        with self.assertRaisesRegex(ValueError,'children'):s.validate_waiting(fixture(),123,[dict(pid=102)])
    def test_stale_running_state_refused(self):
        f=fixture();f['states']['WN']['status']='RUNNING'
        with self.assertRaisesRegex(ValueError,'waiting'):s.validate_waiting(f,123,[])
    def test_wrong_owner_refused(self):
        with self.assertRaisesRegex(ValueError,'owner'):s.validate_waiting(fixture(),456,[])
    def test_report_process_not_coordinator(self):
        f=fixture();f['processes'][0]['cmd']=['python','job20.py','report']
        with self.assertRaisesRegex(ValueError,'coordinator'):s.validate_waiting(f,123,[])
    def test_multiple_coordinators_refused(self):
        f=fixture();f['processes'].append(copy.deepcopy(f['processes'][0]))
        with self.assertRaisesRegex(ValueError,'Multiple'):s.validate_waiting(f,123,[])
    def test_already_stopped_zero_stage_accepted(self):
        f=fixture();f['processes']=[];f['states']['WN']['status']='STOPPED_ERROR';self.assertIsNone(s.validate_waiting(f,123,[]))
    def test_exact_path_matching_never_other_job(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'graphdpo_fb_wn20_full_baselines_v2_job';root.mkdir()
            record=dict(cmd=['python','-B','job20.py','run'],cwd=str(root))
            self.assertEqual(s.resolved_script(record,root),'job20.py')
            record['cwd']=str(Path(t)/'graphdpo_zerofisher_audit_v2_job')
            self.assertIsNone(s.resolved_script(record,root))
            record['cmd']=['python',str(root/'worker20.py'),'session','stage'];self.assertEqual(s.resolved_script(record,root),'worker20.py')
    def test_snapshot_rejects_other_package_without_reading_results(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaisesRegex(ValueError,'exact v2'):s.snapshot(t)
    def test_no_killall_or_sigkill_or_unlink_in_helper(self):
        import ast
        source=(Path(s.__file__)).read_text();tree=ast.parse(source)
        methods=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertNotIn('kill',methods);self.assertNotIn('unlink',methods);self.assertIn('pidfd_send_signal',methods)

    def exercise_stop(self,root,pid_changed=False,already_stopped=False):
        from unittest.mock import Mock
        events=[];f=fixture();f['session']=str(root/'results/old')
        f['processes'][0]['cwd']=str(root)
        if already_stopped:f['processes']=[]
        after=copy.deepcopy(f);after['processes']=[]
        @contextlib.contextmanager
        def guard():
            events.append('lock');yield;events.append('unlock')
        current=dict(pid=101,uid=123,start_ticks=501 if pid_changed else 500,cmd=['python','job20.py','run'],cwd=str(root))
        fake_os=SimpleNamespace(getuid=lambda:123,pidfd_open=Mock(return_value=7),close=Mock())
        fake_signal=SimpleNamespace(SIGTERM=15,pidfd_send_signal=Mock(side_effect=lambda *a:events.append('signal')))
        fake_poll=Mock();fake_poll.poll.return_value=[(7,1)]
        with patch.object(s,'sys',SimpleNamespace(platform='linux')),patch.object(s,'os',fake_os),patch.object(s,'signal',fake_signal),patch.object(s,'select',SimpleNamespace(poll=lambda:fake_poll,POLLIN=1)),patch.object(s,'admission_guard',guard),patch.object(s,'snapshot',side_effect=[f,after]),patch.object(s,'descendants',return_value=[]),patch.object(s,'process_record',return_value=current),patch.object(s,'coordinator_unlocked',side_effect=lambda *a:events.append('coordinator_free')),patch.object(s,'__file__',str(root.parent/'v3/safe_stop_v2.py')):
            (root.parent/'v3').mkdir(exist_ok=True)
            if pid_changed:
                with self.assertRaisesRegex(ValueError,'PID identity'):s.stop_waiting(root)
                fake_signal.pidfd_send_signal.assert_not_called();fake_os.close.assert_called_once_with(7)
            else:s.stop_waiting(root)
        return events,fake_signal,fake_os

    def test_stop_locks_then_pidfd_signals_only_target_then_checks_unlock(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'graphdpo_fb_wn20_full_baselines_v2_job';root.mkdir()
            events,signal,os=self.exercise_stop(root)
            self.assertEqual(events,['lock','signal','coordinator_free','unlock'])
            signal.pidfd_send_signal.assert_called_once_with(7,15,None,0)
            os.pidfd_open.assert_called_once_with(101,0)
            self.assertEqual(list(root.iterdir()),[])
            self.assertEqual(len(list((root.parent/'v3/switch_receipts').glob('*.json'))),1)

    def test_pid_reuse_refuses_signal(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'graphdpo_fb_wn20_full_baselines_v2_job';root.mkdir()
            self.exercise_stop(root,pid_changed=True)

    def test_already_stopped_checks_locks_without_signal(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'graphdpo_fb_wn20_full_baselines_v2_job';root.mkdir()
            events,signal,os=self.exercise_stop(root,already_stopped=True)
            self.assertEqual(events,['lock','coordinator_free','unlock']);signal.pidfd_send_signal.assert_not_called();os.pidfd_open.assert_not_called()
