import copy,hashlib,json,tempfile,time
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch,Mock
import numpy as np
import torch
import schema_inputs as inp,matrix20 as m,job_schema as job,gpu_wait as gpu
from igpu import bootstrap,audit
import common as c,graph_data as gd,rank_kernels as rk,legacy_runtime as rt
from test_schema_adapter import fixture
import worker_schema as worker

def tiny():
    facts,phases=fixture()
    return dict(triples=facts,phases=phases,is_heads=[[False]*5 for _ in phases],designated=[[q[2] for q in qs] for qs in phases],
                counts=dict(entities=18,relations=3,triples=len(facts),per_stage=[5]*4),dataset='FB',name='fb15k-237-10')

class SchemaJob(TestCase):
    def test_roster_no_R_training_no_MetaEU_or_repair(self):
        rows=[x for k in inp.ROSTERS for x in inp.stage_plan(k)]
        self.assertEqual(len(rows),120)
        self.assertEqual(sum(x['mode']=='delete' for x in rows),48)
        self.assertEqual(sum(x['mode']=='evaluate' and x['arm']=='Schema' for x in rows),48)
        self.assertEqual({x['arm'] for x in rows},{'Schema','P'})
        for key in inp.ROSTERS:
            p=inp.stage_plan(key);seen=set()
            for x in p:self.assertTrue(set(x['needs'])<=seen);seen.add(x['name'])
            for x in p:
                if x['mode']=='delete' and x['phase']>0:self.assertEqual(x['previous'],f'train_Schema_{x["seed"]}_p{x["phase"]-1}')

    def test_shared_lock_and_own_OOM(self):
        self.assertEqual(job.LOCK.as_posix(),'/mnt/new4T/lsb/.graphdpo_gpu0_admission.lock')
        w=SimpleNamespace(own_memory_seen=True,competition_seen=False)
        self.assertTrue(gpu.own_baseline_oom(dict(arm='Schema'),w))
        w.competition_seen=True;self.assertFalse(gpu.own_baseline_oom(dict(arm='Schema'),w))
        self.assertEqual(gpu.stage_timeout(dict(mode='delete',arm='Schema')),7200)
        self.assertEqual((gpu.START_GRACE_SECONDS,gpu.POLL_SECONDS),(45,60))

    def test_sklearn_pinned_defaults(self):
        d=worker.sklearn_contract();self.assertEqual(d['effective_n_init'],1);self.assertEqual(d['KMeans']['n_clusters'],3)

    def test_alignment_within_cohorts(self):
        data=tiny();q,nd=inp.canonical(data,1);a=np.column_stack((q,np.ones((len(q),2),dtype='<u4')))
        mix=np.concatenate((a[:nd][::-1],a[nd:][::-1]))
        self.assertTrue(np.array_equal(inp.align(mix,q,nd,18),a))

    def test_cache_cohort_swap_and_invalid_rank_rejected(self):
        data=tiny();q,nd=inp.canonical(data,1);a=np.column_stack((q,np.ones((len(q),2),dtype='<u4')))
        a[[0,nd]]=a[[nd,0]]
        with self.assertRaises(ValueError):inp.align(a,q,nd,18)
        a[[0,nd]]=a[[nd,0]];a[0,4]=0
        with self.assertRaises(ValueError):inp.align(a,q,nd,18)

    def test_A_seal_missing_is_pending_and_wrong_identity_stops(self):
        with tempfile.TemporaryDirectory() as t:
            train,ev=inp.names20(t,'FB',0,0)
            with self.assertRaises(m.Pending):inp.a_source(t,'FB',0,0)
            train.parent.mkdir(parents=True);ev.parent.mkdir(parents=True)
            for s in (train,ev):
                c.write(s.with_name('result.json'),dict(status='RESOURCE_LIMIT'))
                c.write(s,dict(status='COMMITTED',files={'result.json':m.digest(s.with_name('result.json'))}))
            with self.assertRaises(ValueError):inp.a_source(t,'FB',0,0)

    def test_A_import_requires_model_and_query_binding(self):
        with tempfile.TemporaryDirectory() as t:
            train,ev=inp.names20(t,'FB',42,2)
            for s in (train,ev):s.parent.mkdir(parents=True)
            model='1'*64
            r=dict(name=train.parent.name,status='COMPLETE',tag='FB',arm='R',seed=42,phase=2,mode='reference',checkpoint_sha256=model)
            c.write(train.with_name('result.json'),r);c.write(train,dict(status='COMMITTED',files={'result.json':m.digest(train.with_name('result.json')),'final.pt':model}))
            r.update(name=ev.parent.name,mode='evaluate',source_checkpoint_sha256='2'*64,query_sha256='3'*64)
            c.write(ev.with_name('result.json'),r);c.write(ev,dict(status='COMMITTED',files={'result.json':m.digest(ev.with_name('result.json'))}))
            with self.assertRaisesRegex(ValueError,'not bound'):inp.a_source(t,'FB',42,2)

    def test_audit_wait_releases_lock_and_uses300seconds(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for k in inp.ROSTERS:(root/k).mkdir()
            order=[]
            def sleep(n):self.assertEqual(n,300);self.assertIn('release',order);raise RuntimeError('fixture stop')
            with patch.object(job,'pending_audit',return_value=['missing']),patch.object(gpu,'release_before_dependency',side_effect=lambda *a:order.append('release')),patch.object(gpu,'pause',side_effect=sleep):
                with self.assertRaisesRegex(RuntimeError,'fixture stop'):job.run_audits(root,'not-read')
            self.assertEqual(m.read(root/'FB20/gpu_wait_state.json')['status'],'WAIT_DEPENDENCY')

    def test_training_does_not_wait_for_or_open_R(self):
        source=(m.ROOT/'worker_schema.py').read_text()
        self.assertNotIn('a_source(',source);self.assertNotIn('historical_source(',source)
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'stages').mkdir();(root/'stage_logs').mkdir()
            rows=inp.stage_plan('FB10')[:2];c.write(root/'plan.json',dict(stages=rows))
            def fake(cmd,out,log,session,spec,*a,**kw):c.write(out/'result.json',dict(spec,status='COMPLETE'));return 0,False,1
            with patch.object(gpu,'run_stage',side_effect=fake):job.process(root)
            self.assertTrue((root/'gpu_complete.json').is_file())

    def test_CPU_four_request_model_checkpoint_and_evaluation_smoke(self):
        begin=time.monotonic();data=tiny();cfg=m.read(m.ROOT/'configs/P.json');device=torch.device('cpu')
        worker.configure();up,_=worker.tr.model_api(cfg);rt.seed_rng(0)
        model=up.TransE(worker.tr.args_for(cfg,data,0,0,device),gd.kg(data,0))
        source=rt.snapshot_checkpoint(model,rt.new_optimizer(model,1e-4),dict(kind='P',seed=0,phase=-1,updates=0))
        with tempfile.TemporaryDirectory() as t:
            for k in range(4):
                out=Path(t)/str(k);out.mkdir();ev=out/'eval';ev.mkdir()
                result=worker.update(data,cfg,source,out,0,k,device)
                self.assertGreater(result['optimizer_steps'],0);self.assertTrue(result['relation_weights_unchanged'])
                source=rt.load_checkpoint_checked(out/'final.pt',result['checkpoint_sha256'],True)
                worker.evaluate(data,cfg,0,k,device,source,ev,'Schema',extra_preferred=True)
                e=m.read(ev/'evaluation.json');self.assertEqual(e['queries'],2*len(data['triples']));self.assertTrue(e['preferred'])
                raw=np.fromfile(ev/'ranks.u32le',dtype='<u4').reshape(-1,5)
                q,nd=inp.canonical(data,k);self.assertTrue(np.array_equal(raw[:,:3],q));self.assertEqual(e['n_deleted'],nd)
        print('SCHEMA_CPU_SMOKE_SECONDS='+str(round(time.monotonic()-begin,3)))

    def test_audit_identical_to_A_six_roles_and_empty_focus(self):
        data=tiny()
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);folders=[]
            for arm in ('P','R','Schema'):
                for seed in m.SEEDS:
                    f=root/f'{arm}_{seed}';f.mkdir();folders.append(f)
                    c.write(f/'run.json',dict(dataset='FB',method=arm,seed=seed,backbone='TransE',evaluation='full'))
                    for k in range(4):
                        s=f/f'p{k}';s.mkdir();q,nd=inp.canonical(data,k);ranks=np.ones((len(q),2),dtype='<u4')*(2 if arm=='Schema' else 1)
                        np.column_stack((q,ranks)).astype('<u4').tofile(s/'ranks.u32le')
                        c.write(s/'evaluation.json',dict(n_deleted=nd,metrics=rk.summaries(ranks,nd),preferred=[],cohort='full current retained population'))
                    c.write(f/'COMPLETE.json',dict(files=m.signed_payload(f)))
            audit.run(folders,data,root/'audit')
            import csv
            with (root/'audit/cross_bins_by_combination.csv').open(encoding='utf-8-sig') as stream:
                rows=list(csv.DictReader(stream))
            target=[r for r in rows if r['method']=='Schema' and r['phase']=='0' and r['cohort']=='cumulative_forget' and r['direction']=='combined' and r['R_bin']=='all' and r['deletion_phase']=='all']
            self.assertEqual(len(target),6)
            for r in target:self.assertAlmostEqual(float(r['signed']),np.log(2));self.assertAlmostEqual(float(r['abs']),np.log(2))
            empty=[r for r in rows if r['R_bin']=='focus_Pb_le10_Ra_gt100'];self.assertTrue(all(r['signed']=='' for r in empty))

    def test_raw_source_hash_change_refused(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'cache';p.write_bytes(b'changed')
            with self.assertRaises(ValueError):m.verify_file(p,'0'*64)

    def test_Schema_own_OOM_and_timeout_are_terminal_resource(self):
        from test_scheduler20 import Scheduling
        for outcome in ('own','timeout'):
            with tempfile.TemporaryDirectory() as t:
                calls,folder,_=Scheduling().simulate(Path(t),'Schema','delete',[outcome])
                self.assertEqual(calls,[7200]);self.assertEqual(m.read(folder/'result.json')['status'],'RESOURCE_LIMIT')

    def test_memory_wait_releases_lease_and_foreign_process_is_allowed(self):
        from test_scheduler20 import snapshot
        with tempfile.TemporaryDirectory() as t:
            leases=[];events=[]
            def lease(path):
                x=Mock();x.acquire.return_value=True;leases.append(x);return x
            def cooldown(seconds,stop):
                self.assertEqual(seconds,60);leases[-1].release.assert_called_once();events.append('unlocked_wait')
            low=snapshot(foreign=True);low['free_mib']=4095
            high=snapshot(foreign=True);high['free_mib']=4096
            with patch.object(gpu,'StartLease',side_effect=lease),patch.object(gpu,'query',side_effect=[low,high]),patch.object(gpu,'pause',side_effect=cooldown):
                got=gpu.wait_for_gpu(Path(t)/'state.json','Schema',4096)
            self.assertEqual(events,['unlocked_wait']);self.assertIs(got['_lease'],leases[-1]);leases[-1].release.assert_not_called()
            leases[-1].release()

    def test_first_admission_and_measurement_rule(self):
        self.assertEqual(gpu.admission_threshold(None,24564),4096)
        self.assertEqual(gpu.admission_threshold(3000,24564),4500)
        lease=Mock();w=gpu.Watcher(dict(total_mib=24564,free_mib=10000,compute=[],_lease=lease))
        w.started(10);lease.started.assert_called_once_with()
        self.assertEqual(gpu.StartLease.started.__defaults__,(45,))

    def test_retry_anchors_leave_Schema_science_unchanged(self):
        import schema_adapter as a
        from test_schema_adapter import Model
        facts,phases=fixture();torch.manual_seed(42);x=Model();y=copy.deepcopy(x);anchors=[]
        one=a.run_phase(x,facts,phases,0,42)
        two=a.run_phase(y,facts,phases,0,42,on_step=lambda i,m,o:anchors.append(rt.tensor_tree_hash(dict(model=m.state_dict(),rng=rt.capture_rng(),optimizer=o.state_dict()))))
        self.assertEqual(one,two);self.assertTrue(anchors)
        for k,v in x.state_dict().items():self.assertTrue(torch.equal(v,y.state_dict()[k]))

    def test_sealed_A_reference_import_does_not_write_producer(self):
        import audit_schema as au
        data=tiny();query,nd=inp.canonical(data,0)
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);producer=root/'producer';out=root/'own';out.mkdir()
            train,ev=inp.names20(producer,'FB',0,0)
            for p in (train,ev):p.parent.mkdir(parents=True)
            model_sha='a'*64
            c.write(train.with_name('result.json'),dict(tag='FB',arm='R',seed=0,phase=0,mode='reference',name=train.parent.name,status='COMPLETE',checkpoint_sha256=model_sha))
            c.write(train,dict(status='COMMITTED',files={'final.pt':model_sha,'result.json':m.digest(train.with_name('result.json'))}))
            raw=np.column_stack((query,np.ones((len(query),2),dtype='<u4')))
            raw=np.concatenate((raw[:nd][::-1],raw[nd:][::-1])).astype('<u4');raw.tofile(ev.with_name('ranks.u32le'))
            qh=hashlib.sha256(raw[:,:3].tobytes()).hexdigest()
            c.write(ev.with_name('evaluation.json'),dict(method='R',seed=0,phase=0,query_sha256=qh,n_deleted=nd))
            c.write(ev.with_name('result.json'),dict(tag='FB',arm='R',seed=0,phase=0,mode='evaluate',name=ev.parent.name,status='COMPLETE',source_checkpoint_sha256=model_sha,query_sha256=qh))
            job.seal(ev.parent);before=m.signed_payload(producer)
            au.import_reference(data,'FB20','R',0,0,out,producer)
            self.assertEqual(m.signed_payload(producer),before)
            self.assertTrue(np.array_equal(np.fromfile(out/'ranks.u32le',dtype='<u4').reshape(-1,5)[:,:3],query))
            self.assertFalse(train.with_name('final.pt').exists()) # R model bytes are not required or read.

    def test_report_zip_excludes_models_and_ranks(self):
        import contextlib,io,zipfile
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);run=root/'results'/'fixture';run.mkdir(parents=True);c.write(root/'latest_run.json',dict(session=str(run)))
            for key in inp.ROSTERS:
                stage=run/key/'stages'/'train_Schema_0_p0';stage.mkdir(parents=True)
                (stage/'final.pt').write_bytes(b'not a model');(stage/'ranks.u32le').write_bytes(b'not ranks')
                c.write(stage/'result.json',dict(status='COMPLETE'));job.seal(stage)
            with patch.object(job,'ROOT',root),contextlib.redirect_stdout(io.StringIO()):job.inspect(True)
            with zipfile.ZipFile(next(run.glob('report_*.zip'))) as z:
                self.assertFalse(any(n.endswith(('.pt','.u32le')) for n in z.namelist()))
                self.assertIn('OVERALL_STATUS=INCOMPLETE',z.read('report.txt').decode())
