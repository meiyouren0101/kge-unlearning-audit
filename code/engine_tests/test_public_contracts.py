from itertools import permutations
from pathlib import Path
import ast
import hashlib
import json
import unittest

import numpy as np
import torch
from igpu import bootstrap
from audit import score_auc, rebound
from baselines.native_stop import NativeStop
from baselines.zerofisher import select_scale
from scripts.reproduce import ROOT, TABLES, table_plan, settings
import gate_core
import graph_data as gd

class PublicContracts(unittest.TestCase):
    def test_each_table_has_complete_run_roster(self):
        for t in TABLES:
            for graph,backbone,methods in table_plan(t):
                self.assertEqual(methods[:2],('P','R'))
                self.assertEqual(len(set(methods)),len(methods))
                for arm in methods:
                    if arm!='IGPU-calibrated':
                        cfg=json.loads(settings(backbone,arm).read_text(encoding='utf-8'))
                        self.assertEqual(cfg['name'],arm)
                        self.assertEqual(cfg['snapshot_num'],4)
                        self.assertEqual(cfg['pretrain_epochs'],100)
                        self.assertEqual((cfg['batch_size'],cfg['embedding_dim'],cfg['negative_ratio']),(1024,200,10))
    def test_prediction_bytes_match_e1_and_all_text_manifest(self):
        folder=ROOT/'predictions'
        index=json.loads((folder/'E1.json').read_text(encoding='utf-8'))
        self.assertEqual(len(index),12)
        for row in index.values():
            h=hashlib.sha256((folder/row['file']).read_bytes()).hexdigest()
            self.assertEqual(h,row['sha256'])
            self.assertTrue(h.startswith(row['paper_sha256_prefix']))
        for line in (folder/'SHA256SUMS').read_text().splitlines():
            h,name=line.split('  ')
            self.assertEqual(h,hashlib.sha256((folder/name).read_bytes()).hexdigest())
    def test_original_inverse_mask_and_complete_mask(self):
        idx=gd.filters([(0,0,2),(1,0,2)])
        self.assertEqual(idx['legacy'][2,1],{1})
        self.assertEqual(idx['complete'][2,1],{0,1})
    def test_c5_ties_direction_and_empty_classes(self):
        self.assertEqual(score_auc.auc([True,False],[1,1]),.5)
        self.assertIsNone(score_auc.auc([True],[9]))
        self.assertEqual(score_auc.auc([False,True],[1,2]),1)
        qs=np.array([[0,0,1],[1,1,0]],dtype=np.uint32)
        rows=score_auc.evaluate('FB',0,qs,{'S':np.array([1,2])},dict.fromkeys((0,42,123),np.array([1,200])),names=('S',))
        self.assertEqual(len(rows),6)
        self.assertTrue(all(x['AUC'] is None for x in rows))
        self.assertEqual(sum(x['positive_count'] for x in rows),3)
        self.assertEqual(sum(x['negative_count'] for x in rows),3)
    def test_gate_changes_backward_only(self):
        x=torch.tensor([2.,3.],requires_grad=True);w=torch.tensor([.25,1.5])
        y=gate_core.gate(x,w)
        torch.testing.assert_close(x,y,rtol=0,atol=0)
        y.sum().backward();torch.testing.assert_close(x.grad,w,rtol=0,atol=0)
    def test_rebound_subtracts_reference_change_and_does_not_zero_empty(self):
        x=np.array([5.,5.]);y=np.array([2.,10.]);r=np.array([4.,4.]);s=np.array([2.,4.])
        self.assertAlmostEqual(rebound.leak_stat(x,y,r,s,np.array([True,False]))['Leak'],.05)
        self.assertIsNone(rebound.leak_stat(x,y,r,s,np.array([False,False]))['Leak'])
    def test_native_early_stop_tie_does_not_move_checkpoint(self):
        sel=NativeStop()
        self.assertEqual([k+1 for k in range(8) if sel.due(k)],[1,4,7])
        self.assertTrue(sel.observe(0,{'mrr':.2},{'mrr':.2})['improved'])
        self.assertTrue(sel.observe(3,{'mrr':.2},{'mrr':.2})['stop'])
        self.assertEqual(sel.best_epoch,1)
    def test_scale_larger_eta_on_tie_and_no_eligible_default(self):
        rows=[dict(status='COMPLETE',deleted_MRR=.1,retained_MRR=.95,scale=s) for s in (.1,3,30)]
        self.assertEqual(select_scale(rows,1),(30,False))
        self.assertEqual(select_scale(rows,1.1),(10,True))
    def test_compatible_python_310_source(self):
        for p in ROOT.rglob('*.py'):
            if not {'.upstream','.venv','outputs'} & set(p.relative_to(ROOT).parts):
                ast.parse(p.read_text(encoding='utf-8'),feature_version=(3,10))
