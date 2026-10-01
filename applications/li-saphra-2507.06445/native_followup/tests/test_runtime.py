"""Operator tests, independent of any desired empirical effect."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import DyckRuntime, factorized_eos_weights, make_model, tokenize


class FactorizationTests(unittest.TestCase):
    def setUp(self):
        self.a = torch.tensor([[.1, .2, .6, .1, 0.], [.3, .2, .5, 0., 0.]], dtype=torch.float64)
        self.eos = torch.tensor([3, 2])

    def test_routing_preserves_each_endpoint_and_bracket_mass(self):
        actual = factorized_eos_weights(self.a, self.eos, "routing_only")
        expected = torch.tensor([[.1, .4, .4, .1, 0.], [.3, .2, .5, 0., 0.]], dtype=torch.float64)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_gate_preserves_relative_bracket_routing(self):
        actual = factorized_eos_weights(self.a, self.eos, "gate_only")
        expected = torch.tensor([[.25, .125, .375, .25, 0.], [1/3, 1/3, 1/3, 0., 0.]], dtype=torch.float64)
        torch.testing.assert_close(actual, expected, atol=1e-15, rtol=0)
        self.assertAlmostEqual(float(actual[0, 1] / actual[0, 2]), 1/3)

    def test_both_uniform_and_two_orders_of_factorization_agree(self):
        both = factorized_eos_weights(self.a, self.eos, "both")
        for first, second in [("routing_only", "gate_only"), ("gate_only", "routing_only")]:
            sequential = factorized_eos_weights(factorized_eos_weights(self.a, self.eos, first), self.eos, second)
            torch.testing.assert_close(sequential, both, atol=1e-15, rtol=0)

    def test_invalid_weight_rows_are_rejected(self):
        for bad in [torch.tensor([[.2, .2, .6, .1, 0.]]), torch.tensor([[.2, -.2, .6, .4, 0.]]),
                    torch.tensor([[.2, .2, .3, .2, .1]])]:
            with self.assertRaises(ValueError):
                factorized_eos_weights(bad, self.eos[:1], "both")

    def test_undefined_native_routing_is_not_silently_clamped(self):
        with self.assertRaisesRegex(ValueError, "routing is undefined"):
            factorized_eos_weights(torch.tensor([[.5, 0., 0., .5]]), self.eos[:1], "gate_only")

    def test_tokenization(self):
        token, eos = tokenize(["()", ")()("])
        self.assertEqual(token[0, :6].tolist(), [0, 3, 4, 2, 1, 1])
        self.assertEqual(eos.tolist(), [3, 5])
        for bad in [[], [""], ["a"], ["(" * 41]]:
            with self.assertRaises(ValueError):
                tokenize(bad)


class ForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(812)
        cls.runtime = DyckRuntime(make_model(n_embd=16), layer=1, head=1)
        cls.inputs = ["()", ")(", "((()))", ")()(", "(())()"]

    def test_reconstruction_is_bitwise_identical_to_upstream(self):
        native = self.runtime.run(self.inputs)
        identity = self.runtime.run(self.inputs, mode="identity")
        for field in ["margins", "eos_logits", "attention", "full_attention", "values", "node", "preprojection"]:
            self.assertTrue(torch.equal(getattr(native, field), getattr(identity, field)), field)

    def test_only_selected_eos_head_slice_changes(self):
        native = self.runtime.run(self.inputs)
        for mode in ["routing_only", "gate_only", "both"]:
            changed = self.runtime.run(self.inputs, mode=mode)
            mask = torch.ones_like(native.preprojection, dtype=torch.bool)
            index = torch.arange(len(self.inputs))
            mask[index, native.eos_positions, 8:16] = False
            self.assertTrue(torch.equal(native.preprojection[mask], changed.preprojection[mask]))
            self.assertTrue(torch.equal(native.values, changed.values))
            self.assertTrue(torch.any(native.node != changed.node))

    def test_eos_only_and_all_query_uniform_have_same_final_readout(self):
        all_query = self.runtime.run(self.inputs, mode="uniform_all_queries")
        eos_only = self.runtime.run(self.inputs, mode="both")
        # This equivalence is specific to intervening in the last attention
        # layer: later computations cannot carry other query changes to EOS.
        torch.testing.assert_close(all_query.eos_logits, eos_only.eos_logits, atol=2e-7, rtol=1e-6)
        self.assertTrue(torch.equal(all_query.predicted_valid, eos_only.predicted_valid))

    def test_custom_attention_and_node_identity(self):
        native = self.runtime.run(self.inputs)
        for mode, arg in [("custom_attention", {"attention_override": native.attention}),
                          ("custom_node", {"node_override": native.node})]:
            actual = self.runtime.run(self.inputs, mode=mode, **arg)
            self.assertTrue(torch.equal(native.eos_logits, actual.eos_logits))
        actual = self.runtime.run(self.inputs, mode="custom_attention",
                                  attention_override=lambda a, v, eos: a)
        self.assertTrue(torch.equal(native.eos_logits, actual.eos_logits))

    def test_cleanup_after_callback_exception(self):
        module = self.runtime.module
        before = self.runtime.run(self.inputs).eos_logits
        def fail(*args):
            raise RuntimeError("deliberate")
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            self.runtime.run(self.inputs, mode="custom_attention", attention_override=fail)
        self.assertNotIn("forward", module.__dict__)
        self.assertFalse(any(module.ablate_heads))
        self.assertEqual(len(module.c_attn._forward_hooks), 0)
        self.assertEqual(len(module.c_proj._forward_pre_hooks), 0)
        self.assertTrue(torch.equal(before, self.runtime.run(self.inputs).eos_logits))

    def test_calls_and_input_order_with_multiple_batches(self):
        result = self.runtime.run(self.inputs, batch_size=2, mode="both")
        self.assertEqual(result.forward_calls, 3)
        self.assertEqual(result.eos_positions.tolist(), [3, 3, 7, 5, 7])
        single_batch = self.runtime.run(self.inputs, batch_size=8, mode="both")
        torch.testing.assert_close(result.eos_logits, single_batch.eos_logits, atol=2e-7, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
