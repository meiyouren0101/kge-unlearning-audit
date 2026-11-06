from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
from itertools import permutations
import hashlib, io, json, tempfile, zipfile
import numpy as np
import torch
import job20, matrix20 as m, worker20 as w, gpu_wait as gpu
from igpu import training as tr
import graph_data as gd, legacy_runtime as rt

class Matrix(TestCase):
    def test_order_and_roster(self):
        for tag in m.SPECS:
            for policy in ('grid','transfer'):
                rows=m.matrix(tag,policy);names=set()
                self.assertEqual(len(rows),337)
                for r in rows:
                    self.assertTrue(set(r['needs']) <= names); self.assertNotIn(r['name'],names); names.add(r['name'])
                self.assertEqual(len([r for r in rows if r['mode']=='evaluate']),168)
                self.assertEqual(len([r for r in rows if r['mode']=='reference']),12)
                self.assertEqual(len([r for r in rows if r['mode']=='repair']),12)
                self.assertEqual(len({r['arm'] for r in rows if r['mode']=='delete'}),12)
                self.assertFalse(any(r['arm']=='P' and r['mode']!='evaluate' for r in rows))
    def test_fixed_hyperparameters(self):
        for arm in ('P','R',*m.METHODS):
            c=w.config(arm)
            self.assertEqual((c['pretrain_epochs'],c['delete_epochs'],c['batch_size'],c['margin'],c['negative_ratio']), (100,20,1024,8.,10))
            self.assertEqual(c['delete_lr'],.0001 if arm=='FT-lr1e-4' else .001)
        self.assertEqual(w.config('GraphDPO-code')['objective'],'code')
        self.assertEqual(w.config('GraphDPO')['objective'],'paper')
        self.assertEqual(w.config('IGPU')['gate_scale'],1)
        self.assertTrue(w.config('IGPU')['repair'] and w.config('IGPU')['real_targets'])
        self.assertTrue(w.config('Fisher')['fisher_stream'])
    def test_p_pins_all_seed_blocks(self):
        rows=m.read(m.ROOT/'P_PINS.json')
        for tag in m.SPECS:
            self.assertEqual(set(rows[tag]),{'0','42','123'})
            self.assertTrue(all(len(p['sha256'])==64 for p in rows[tag].values()))
    def test_toy_archive_5th_not_used(self):
        b=io.BytesIO();raw=b'a\tr\tb\nc\tr\td\nb\tr\tc\nd\tr\ta\na\tr\td\nb\tr\ta\n'
        with zipfile.ZipFile(b,'w') as z:
            z.writestr('toy/all_data.txt',raw)
            req=['a\tr\tb\tb','c\tr\td\td','b\tr\tc\tc','d\tr\ta\ta','intentionally_not_a_parsed_request']
            for i,r in enumerate(req):z.writestr(f'toy/{i}/{i}.txt',r+'\n')
        zipbytes=b.getvalue();n=len(zipbytes)
        spec=('toy',n,hashlib.sha1(f'blob {n}\0'.encode()+zipbytes).hexdigest(),hashlib.sha256(raw).hexdigest(),4,1,6,1)
        with tempfile.TemporaryDirectory() as t,patch.dict(m.SPECS,{'FB':spec}):
            p=Path(t)/'a.zip';p.write_bytes(zipbytes);d=m.graph('FB',p)
            self.assertEqual([len(gd.at(d,k)['retained']) for k in range(4)],[5,4,3,2])
            p.write_bytes(zipbytes[:-1]+b'x')
            with self.assertRaisesRegex(ValueError,'pinned official'):m.graph('FB',p)
    def test_scale_full_order_and_fallback(self):
        a=[dict(status='COMPLETE',scale=x,retained_MRR=.95,deleted_MRR=.1) for x in m.SCALES]
        self.assertEqual(len(a),13)
        self.assertEqual(m.select_scale(a,1),(100,False))
        self.assertEqual(m.select_scale(a,1.01),(10,True))
    def test_admission_foreign_is_not_old_self(self):
        from unittest.mock import Mock
        lease=Mock();lease.acquire.return_value=True
        st=dict(total_mib=24564,free_mib=5000,compute=[dict(pid=77,used_mib=18000)],time='NOW')
        with patch.object(gpu,'StartLease',return_value=lease),patch.object(gpu,'query',return_value=st),patch.object(gpu,'write_state'):
            got=gpu.wait_for_gpu(Path('unused'),'FB/train',min_free_mib=4096)
        self.assertIs(got['_lease'],lease)
        lease.release.assert_not_called()
    def test_low_memory_release_before_sixty_second_wait(self):
        from unittest.mock import Mock
        lease=Mock();lease.acquire.return_value=True
        st=dict(total_mib=24564,free_mib=4000,compute=[],time='NOW')
        def pause(seconds,stop):
            lease.release.assert_called_once()
            self.assertEqual(seconds,60)
            raise RuntimeError('end fixture')
        with patch.object(gpu,'StartLease',return_value=lease),patch.object(gpu,'query',return_value=st),patch.object(gpu,'write_state'),patch.object(gpu,'pause',side_effect=pause):
            with self.assertRaisesRegex(RuntimeError,'end fixture'):gpu.wait_for_gpu(Path('unused'),'FB/train',min_free_mib=4096)
    def test_reference_rank_cpu_numeric(self):
        from types import SimpleNamespace
        cfg=dict(w.config('P'),embedding_dim=4,pretrain_epochs=1)
        triples=[(i,0,(i+1)%7) for i in range(7)]+[(0,2,5),(1,2,5)]
        d=dict(dataset='FB',name='fb15k-237-20',counts=dict(entities=7,relations=2,triples=len(triples),per_stage=[1]*4),
               triples=triples,phases=[[q] for q in triples[:4]],is_heads=[[False] for _ in range(4)],designated=[[q[2]] for q in triples[:4]])
        device=torch.device('cpu')
        p=tr.pretrain(d,cfg,0,device,phase=0)
        self.assertEqual(p['metadata']['updates'],1)
        model=tr.load_model(d,cfg,0,0,device,p);d0=gd.at(d,0)
        qs=list(gd.queries(d0['deleted_order']))+list(gd.queries(d0['retained']))
        f=io.BytesIO();r=w.ranks(model,qs,gd.filters(d['triples']),device,4,stream=f)
        np.testing.assert_array_equal(r[:,:3],qs)
        self.assertEqual(f.getvalue(),r.tobytes())
        self.assertTrue(((r[:,3:]>=1)&(r[:,3:]<=7)).all())
    def test_pinned_upstream_and_advertorch(self):
        up,_=tr.model_api(w.config('ADV-IMP'),True)
        self.assertEqual(up.L2PGDAttack.__name__,'L2PGDAttack')
        self.assertTrue(all(hasattr(up,n) for n in ('FisherTrainBatchProcessor','BSTrainBatchProcessor','SSDTrainBatchProcessor','NGTrainBatchProcessor','RLTrainBatchProcessor')))
