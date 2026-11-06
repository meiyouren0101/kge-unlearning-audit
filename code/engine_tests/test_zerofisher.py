import json
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from igpu import bootstrap
from baselines import author_adapter, zf_core as zf
import graph_data as gd
import legacy_runtime as rt
import official_api
import rank_kernels as rk


class MathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.up, _ = official_api.active()

    def model(self, n=8, dim=200):
        rt.seed_rng(42,False)
        args=SimpleNamespace(device=torch.device('cpu'),emb_dim=dim,margin=8,neg_ratio=10)
        return self.up.TransE(args,SimpleNamespace(ent_num=n,rel_num=4))

    def test_woodbury_matches_dense_inverse(self):
        gen=torch.Generator().manual_seed(27)
        v,g=[torch.randn(8,generator=gen,dtype=torch.float64) for _ in range(2)]
        expected=torch.linalg.solve(torch.eye(8,dtype=g.dtype)+torch.outer(g,g),v)
        actual=torch.cat(zf.ihvp((v[:4],v[4:]),(g[:4],g[4:])))
        torch.testing.assert_close(actual,expected,rtol=1e-14,atol=1e-14)

    def test_radius_is_one_hop_and_includes_inverse_relation_rows(self):
        self.assertEqual(zf.edit_rows([(0,0,1)],[(2,2,0),(2,2,3),(1,2,4)]),([0,1,2,4],[0,1]))

    def test_exact_weighted_batches_equal_joint_gradient(self):
        model=self.model()
        facts=[(0,0,1),(2,2,3),(4,2,5)]
        ds=self.up.TrainDataset(model.args,SimpleNamespace(train_data=facts,ent_num=8))
        samples=[ds[i] for i in range(len(ds))]
        collate=ds.collate_fn
        rows=tuple(torch.arange(len(p)) for p in model.embedding())
        exact=zf.batch_gradient(model,collate(samples),rows,None,True)
        result,n=zf.cohort_gradient(model,[collate(samples[:4]),collate(samples[4:])],rows,None,True,'f')
        self.assertEqual(n,66)
        for actual,expected in zip(result,exact):
            torch.testing.assert_close(actual,expected,rtol=1e-13,atol=1e-13)

    def test_zo_matches_directional_exact_gradient_and_restores(self):
        model=self.model()
        ds=self.up.TrainDataset(model.args,SimpleNamespace(train_data=[(0,0,1),(2,2,3)],ent_num=8))
        batch=ds.collate_fn([ds[i] for i in range(len(ds))])
        rows=(torch.tensor([0,1,2,3]),torch.tensor([0,1,2,3]))
        original=rt.tensor_tree_hash(model.state_dict())
        noise=torch.Generator().manual_seed(11)
        u=tuple(torch.randn((len(r),200),generator=noise,dtype=torch.float64) for r in rows)
        true=zf.batch_gradient(model,batch,rows,None,True)
        slope=zf.inner(true,u)
        actual=zf.batch_gradient(model,batch,rows,noise.manual_seed(11),False)
        for g,direction in zip(actual,u):
            torch.testing.assert_close(g,slope*direction,rtol=1e-7,atol=1e-9)
        self.assertEqual(original,rt.tensor_tree_hash(model.state_dict()))
        self.assertFalse(torch.equal(u[0][0],u[0][1]))

    def test_main_arms_full_inputs_and_replay(self):
        for exact in (False,True):
            m=self.model()
            graph=dict(triples=[(0,0,1),(2,2,0),(2,2,3),(1,0,4),(6,2,7)],
                       phases=[[(0,0,1)]],counts={'entities':8})
            old=tuple(p.detach().clone() for p in m.embedding())
            ids,step,info=zf.correction(self.up,m,graph,0,19,exact)
            ids2,step2,_=zf.correction(self.up,m,graph,0,19,exact)
            self.assertEqual(info['deleted_loss_rows'],22)
            self.assertEqual(info['retained_loss_rows'],88)
            self.assertEqual(rt.tensor_tree_hash((ids,step)),rt.tensor_tree_hash((ids2,step2)))
            zf.apply(m,ids,step,10)
            for before,after,rs in zip(old,m.embedding(),ids):
                untouched=sorted(set(range(len(before)))-set(rs.tolist()))
                self.assertTrue(torch.equal(before[untouched],after[untouched]))
            self.assertGreater(float((m.embedding()[0]-old[0]).norm()),0)

    def test_strongest_candidate_lowers_deleted_mrr_with_average_gradients(self):
        graph=dict(triples=[(0,0,1),(2,2,0),(2,2,3),(1,0,4),(6,2,7)],
                   phases=[[(0,0,1)]],counts={'entities':8})
        queries=list(gd.queries(graph['phases'][0]))
        index=gd.filters(graph['triples'])
        self.assertEqual(min(zf.SCALES),1e-4)
        for exact in (False,True):
            model=self.model()
            with torch.no_grad():
                head=model.ent_norm(model.ent_embeddings.weight[0])
                rel=model.rel_norm(model.rel_embeddings.weight[0])
                model.ent_embeddings.weight[1].copy_(head+rel)

            def deleted_mrr():
                with torch.no_grad():
                    ranks=rk.score(torch,model,queries,index,model.args.device,('complete',))['complete']
                return sum(1/r for r in ranks)/len(ranks)

            before=deleted_mrr()
            ids,step,info=zf.correction(self.up,model,graph,0,19,exact)
            zf.apply(model,ids,step,min(zf.SCALES))
            after=deleted_mrr()
            self.assertEqual((info['deleted_loss_rows'],info['retained_loss_rows']),(22,88))
            self.assertLess(after,before)
            print(f'STRONGEST_SCALE_CPU exact={exact} MRR={before:.9f}->{after:.9f}',flush=True)

    def test_author_broadcast_and_first_batches(self):
        m=self.model()
        raw=dict(batch_h=np.array([0,2,1]),batch_t=np.array([1,3,4]),
                 batch_r=np.array([0,2,3]),batch_y=np.array([1,1,1]),mode='normal')
        second=dict(raw,batch_t=np.array([6,0,2]))
        class FirstOnly:
            def __init__(self,data):
                self.data=data
            def __iter__(self):
                yield self.data
                raise AssertionError('Official read a second batch')
        before=tuple(p.detach().clone() for p in m.embedding())
        if not author_adapter.PINS.is_file() or not author_adapter.SOURCE.is_dir():
            self.skipTest('Optional pinned ZOWFKGIF external source is not staged')
        result=author_adapter.apply(m,FirstOnly(raw),FirstOnly(second),cpu_fixture=True)
        self.assertEqual(result['score_calls'],8)
        self.assertTrue(all(0<n<1 for n in result['row_update_l2']))
        for p,old in zip(m.embedding(),before):
            delta=p.detach()-old
            torch.testing.assert_close(delta,delta[:1].expand_as(delta),rtol=0,atol=2e-16)
        print('OFFICIAL_BROADCAST_CPU',json.dumps(result),flush=True)

