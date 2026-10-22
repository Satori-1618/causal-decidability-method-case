"""Operator checks on seeded synthetic models only; no released checkpoint."""
import json
from pathlib import Path
import sys
import unittest

import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parent/'native_followup'))
from runtime import DyckRuntime,make_model
from value_runtime import run_value_transfers


class ValueTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(781)
        cls.rt=DyckRuntime(make_model(n_layer=2,n_head=2,n_embd=16,dtype=torch.float64))

    def test_operator_formula_and_snapshot(self):
        out=run_value_transfers(self.rt,['(())','()()'],['()()','(())'],[2,3],[1,2])
        self.assertTrue(torch.equal(out.h_patch_intended,out.h_r+out.a_r[:,None]*(out.v_d-out.v_r)))
        self.assertTrue(torch.equal(out.h_patch_delivered,out.h_patch_intended))
        self.assertTrue(torch.any(out.node_delta_l2>0))
        self.assertTrue(out.controls['all_target_layer_attention_weights_unchanged'])
        self.assertTrue(out.controls['all_target_layer_value_projections_unchanged'])
        self.assertEqual(out.controls['identity_max_margin_error'],0)
        snapshot=out.snapshots()
        json.dumps(snapshot,allow_nan=False)
        self.assertEqual(snapshot[0]['recipient_position'],2)
        self.assertEqual(snapshot[0]['donor_position'],1)
        self.assertEqual(len(snapshot[0]['v_d']),8)

    def test_self_same_position_is_exact(self):
        out=run_value_transfers(self.rt,['()','(())'],['()','(())'],[1,3],[1,3])
        self.assertTrue(torch.equal(out.patched_margins,out.native_margins))
        self.assertEqual(int(torch.count_nonzero(out.requested_node_delta)),0)
        self.assertEqual(out.controls['self_same_position_cases'],2)
        self.assertEqual(out.controls['self_same_position_max_margin_error'],0)

    def test_variable_lengths_and_multiple_batches_keep_pair_order(self):
        recipients=['()','(())','()()']
        donors=['()','()()','(())']
        a=run_value_transfers(self.rt,recipients,donors,[1,2,3],[1,1,2],batch_size=1)
        b=run_value_transfers(self.rt,recipients,donors,[1,2,3],[1,1,2],batch_size=3)
        torch.testing.assert_close(a.patched_margins,b.patched_margins,atol=1e-12,rtol=0)
        self.assertEqual(a.controls['transfer_forward_batches'],3)
        self.assertEqual(a.recipient_positions.tolist(),[1,2,3])

    def test_invalid_pairs_fail_before_forward(self):
        cases=[([],[],[],[]),(['()'],['(())'],[1],[1]),(['()'],['()'],[0],[1]),
               (['()'],['()'],[3],[1]),(['()'],['()'],[1],[2]),(['()'],['()'],[1.],[1]),
               (['()'],['()'],[True],[1]),(['()'],['()'],[[1]],[1]),(['xx'],['()'],[1],[1])]
        for args in cases:
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    run_value_transfers(self.rt,*args)

    def test_nonfinal_layer_rejected(self):
        early=DyckRuntime(self.rt.model,layer=0,head=1)
        with self.assertRaisesRegex(ValueError,'final attention layer'):
            run_value_transfers(early,['()'],['()'],[1],[1])

    def test_external_capture_hooks_are_removed(self):
        run_value_transfers(self.rt,['(())'],['()()'],[2],[1])
        self.assertEqual(len(self.rt.module._forward_hooks),0)
        self.assertEqual(len(self.rt.module.c_attn._forward_hooks),0)

    def test_float32_operator_and_identity(self):
        torch.manual_seed(782)
        rt=DyckRuntime(make_model(n_layer=2,n_head=2,n_embd=16,dtype=torch.float32))
        out=run_value_transfers(rt,['(())','()()'],['()()','()()'],[2,3],[1,3])
        self.assertTrue(out.controls['inserted_node_exactly_intended_after_dtype'])
        self.assertEqual(out.controls['self_same_position_max_margin_error'],0)
        self.assertLessEqual(out.controls['identity_max_margin_error'],1e-5)


if __name__=='__main__':
    unittest.main()
