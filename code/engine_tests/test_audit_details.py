import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,tempfile,unittest
import numpy as np
from igpu import bootstrap
from igpu import training as tr,audit
import probe_kernels as pk,case_kernels as ck,sampled_metrics,absolute_metrics as am,bootstrap_core as bc

class AuditDetails(unittest.TestCase):
    def test_original_true_answer_request_weighting(self):
        plan=dict(truth_owners=np.array([[0,0],[0,1],[1,2]]),independent=np.array([[0,0,1],[1,1,2]]))
        rows=pk.truth_request_metrics([1,10,2],plan)
        self.assertEqual(rows[0]['count'],2);self.assertEqual(rows[0]['MRR'],(.55+.5)/2)
    def test_sampled_directions_follow_relation_ids_not_row_parity(self):
        qs=[(0,0,1),(1,2,0),(0,1,2)]
        rows=sampled_metrics.summaries([(1,1),(2,2),(3,3)],2,qs)
        x=next(r for r in rows if (r['filter'],r['cohort'],r['direction'])==('complete','cumulative_forget','inverse'))
        self.assertEqual(x['count'],0);self.assertIsNone(x['MRR'])
    def test_original_repair_budgets_and_round_before_scale(self):
        self.assertEqual([tr.repair_budget(93003-(k+1)*9300) for k in range(4)],[390,347,305,259])
        self.assertEqual([tr.repair_budget(310116-(k+1)*31011) for k in range(4)],[1299,1154,1009,866])
        self.assertEqual(tr.repair_budget(25*512),60);self.assertEqual(tr.repair_budget(25*512,(1,2)),30)
    def test_data_source_and_prepared_manifests_match(self):
        root=Path(__file__).resolve().parents[1]
        pins=json.loads((root/'igpu/assets/SOURCE_PINS.json').read_text('utf-8'))
        prepared=json.loads((root/'igpu/assets/PREPARED_SHA256.json').read_text('utf-8'))
        self.assertEqual(set(prepared),{'FB','WN','CL','YA'})
        for tag,files in prepared.items():
            self.assertEqual(set(files),{'all_data.txt',*(f'{k}/{k}.txt' for k in range(5))})
            self.assertEqual(files['all_data.txt'],pins['all_data_sha256'][tag])
    def test_case_thresholds_not_relaxed_to_fill_five(self):
        row=dict(seed=0,fact_id=0,direction='forward',preferred_unobserved=True,P_preferred_rank=100)
        row.update({m+'_rank':v for m,v in zip(ck.METHODS,(5,20,110,150))});row.update({m+'_preferred_rank':v for m,v in zip(ck.METHODS,(100,100,100,1))})
        first,second,counts=ck.select([row]);self.assertEqual(len(first),1);self.assertEqual(len(second),0)
        row['M_rank']=31;first,second,counts=ck.select([row]);self.assertEqual(len(first),0)
    def test_fixed_fact_bootstrap_shared_weight_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            col=bc.Columns(tmp,2)
            ids=np.array([0,0,1,1]);a=col.add('a',ids,np.array([1.,3.,2.,2.]));den=col.add('n',ids,np.ones(4));b=col.add('b',ids,np.array([0.,2.,1.,1.]))
            matrix=col.close();groups=[dict(point=1.,components=[(a,den,b,den)]*6)]
            result,digest=bc.resample(matrix,groups,8,123)
            np.testing.assert_array_equal(result,np.ones((8,1)));matrix._mmap.close()
    def test_RR_absolute_noise_does_not_cancel(self):
        a=np.array([1.,10.]);b=np.array([10.,1.]);mask=np.ones(2,bool)
        left,_=am.summarize(a,b,mask);right,_=am.summarize(b,a,mask)
        self.assertEqual(left['signed_log_mean_secondary']+right['signed_log_mean_secondary'],0)
        self.assertEqual(left['abs_log_mean'],right['abs_log_mean']);self.assertGreater(left['abs_log_mean'],0)
    def test_report_rejects_modified_rank_cache(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'x.u32le').write_bytes(b'original');(root/'run.json').write_text('{}')
            (root/'COMPLETE.json').write_text(json.dumps(dict(files={'x.u32le':hashlib.sha256(b'original').hexdigest()})))
            (root/'x.u32le').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Changed'):audit.verify(root)
if __name__=='__main__':unittest.main(verbosity=2)
