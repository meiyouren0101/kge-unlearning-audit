import copy, json, os, random, time
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
import numpy as np
import torch
import schema_adapter as a

class Model(torch.nn.Module):
    def __init__(self,n=18):
        super().__init__();self.ent_embeddings=torch.nn.Embedding(n,200,dtype=torch.float64);self.rel_embeddings=torch.nn.Embedding(6,200,dtype=torch.float64)

def fixture():
    # Dense retained neighborhoods ensure several real optimizer updates.
    facts=[(h,2*((h+t)%3),t) for h in range(1,18) for t in range(1,18) if h!=t]
    requests=[facts[k*5:(k+1)*5] for k in range(4)]
    return facts,requests

class SchemaReview(TestCase):
    def test_source_pin(self):self.assertTrue(a.upstream_class())

    def test_retained_only_no_future_leakage(self):
        facts,requests=fixture()
        for k in range(4):
            retained,current=a.stage_lists(facts,requests,k)
            expected=set(facts)-set(f for r in requests[:k+1] for f in r)
            self.assertEqual({(h,r,t) for h,t,r in retained},expected)
            self.assertEqual(current,[(h,t,r) for h,r,t in requests[k]])
            for future in requests[k+1:]:self.assertTrue(set(future)<=expected)

    def test_adjacency_has_no_deleted_only_edges(self):
        cls=a.upstream_class();adj=cls.create_adj_matrix(None,[(1,3,0),(2,3,0)])
        self.assertNotIn(2,adj[1]);self.assertEqual(adj[1][3],0)

    def test_original_ast_three_seed_exact_endpoints(self):
        facts,requests=fixture();rows=[]
        for seed in (0,42,123):
            torch.manual_seed(seed);model=Model();original=copy.deepcopy(model)
            retained,current=a.stage_lists(facts,requests,0)
            loader=a.loader_for(original,retained,current)
            loss=a.reference_original_loop(original,loader,seed)
            result=a.run_phase(model,facts,requests,0,seed)
            maximum=max(float((model.state_dict()[k]-v).abs().max()) for k,v in original.state_dict().items())
            self.assertGreater(result['updates'],0);self.assertEqual(maximum,0);self.assertAlmostEqual(sum(result['losses']),loss,places=14)
            rows.append(dict(seed=seed,max_model_error=maximum,updates=result['updates'],loss_sum_error=abs(sum(result['losses'])-loss)))
        print('SCHEMA_EXACT_COMPARISON='+json.dumps(rows))

    def test_relation_index_same_rng_matches_and_weights(self):
        facts,requests=fixture();torch.manual_seed(42);x=Model();y=copy.deepcopy(x)
        p=a.run_phase(x,facts,requests,0,42,indexed=True);q=a.run_phase(y,facts,requests,0,42,indexed=False)
        self.assertEqual(p,q)
        for k,v in x.state_dict().items():self.assertTrue(torch.equal(v,y.state_dict()[k]))

    def test_four_request_inheritance_and_no_relation_update(self):
        facts,requests=fixture();torch.manual_seed(0);x=Model();rel=x.rel_embeddings.weight.detach().clone();rows=[]
        for k in range(4):
            before=x.ent_embeddings.weight.detach().clone();r=a.run_phase(x,facts,requests,k,0);rows.append(r)
            self.assertGreater(r['updates'],0);self.assertFalse(torch.equal(before,x.ent_embeddings.weight))
            self.assertTrue(torch.equal(rel,x.rel_embeddings.weight));self.assertEqual(r['actual_lr'],1e-5)
        torch.manual_seed(0);fresh=Model();a.run_phase(fresh,facts,requests,3,0)
        self.assertFalse(torch.equal(fresh.ent_embeddings.weight,x.ent_embeddings.weight))
        self.assertEqual(len(rows),4)

    def test_no_match_skips_without_fabrication(self):
        facts=[(0,0,1),(2,2,3),(4,4,5),(6,6,7)];requests=[[f] for f in facts];x=Model();old=copy.deepcopy(x.state_dict())
        r=a.run_phase(x,facts,requests,0,0)
        self.assertEqual((r['updates'],r['skipped']),(0,1))
        for k,v in old.items():self.assertTrue(torch.equal(v,x.state_dict()[k]))

    def test_upstream_zero_id_truthiness_is_preserved(self):
        cls=a.upstream_class();obj=cls.__new__(cls)
        triples=[(0,3,0),(4,5,0)];adj={1:{6:0},2:{7:0},0:{8:0},3:{9:0},4:{8:0},5:{9:0}}
        result=obj.query_match_entity(triples,adj,(1,2,0),[0]*18)
        # Original chooses head 0 then treats it as false; no silent bug fix.
        self.assertIsNone(result[2]);self.assertIsNone(result[3])

    def test_duplicate_or_nontruth_requests_rejected(self):
        facts,requests=fixture();requests[1]=requests[0]
        with self.assertRaises(ValueError):a.stage_lists(facts,requests,1)

    def test_FB_and_WN_same_defaults(self):
        import ast
        tree=ast.parse((a.UP/'Exp_FB15K237/Entity_TransE_FB15K237.py').read_text())
        defaults={node.targets[0].id:ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('lr','n_cluster')}
        self.assertEqual(defaults,dict(lr=1e-5,n_cluster=3))
        # No graph-dependent hyperparameter branch exists in the adapter.
        source=(a.ROOT/'schema_adapter.py').read_text();self.assertNotIn("if dataset",source)
