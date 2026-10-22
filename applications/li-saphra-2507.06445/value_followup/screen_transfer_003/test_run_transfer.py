"""Runner bookkeeping tests use only fake arrays and fake forwards, never torch."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import run_transfer as runner


class Array:
    def __init__(self, values):
        self.values = values
    def tolist(self):
        return self.values
    def __getitem__(self, index):
        return Array(self.values[index])


class FakeModel:
    def __init__(self):
        self.before, self.after = [], []
    def register_forward_pre_hook(self, fn):
        self.before.append(fn)
        return SimpleNamespace(remove=lambda: self.before.remove(fn))
    def register_forward_hook(self, fn):
        self.after.append(fn)
        return SimpleNamespace(remove=lambda: self.after.remove(fn))
    def call(self, n, fail=False):
        args = (SimpleNamespace(shape=(n, 42)),)
        for fn in self.before:
            fn(self, args)
        if fail:
            raise RuntimeError('Synthetic forward failure')
        for fn in self.after:
            fn(self, args, None)


class FakeRuntime:
    def __init__(self, margins):
        self.model = FakeModel()
        self.margins = margins
    def run(self, strings):
        self.model.call(len(strings))
        attention = [[0.] * 42 for _ in strings]
        for row in attention:
            row[2] = .3
        return SimpleNamespace(margins=Array(self.margins), attention=Array(attention))


class RunnerTests(unittest.TestCase):
    def test_import_does_not_load_torch(self):
        result = subprocess.run([sys.executable, '-c',
            'import run_transfer,sys; assert "torch" not in sys.modules'], cwd=runner.HERE,
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_pending_release_blocks_before_output_or_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'forbidden'
            # Force a pending fixture even after a future real release. This test
            # must never accidentally become an authorized inference command.
            script = '''
import json,sys
from pathlib import Path
import run_transfer as r
original=Path.read_text
def read(path,*args,**kwargs):
    text=original(path,*args,**kwargs)
    if path==r.HERE/'plan.json':
        value=json.loads(text);value['status']='awaiting_final_pre_run_review';value['execution_authorized']=False
        return json.dumps(value)
    return text
Path.read_text=read
sys.argv=['run_transfer.py','--execute','--output',sys.argv[1]]
try:r.main()
finally:assert 'torch' not in sys.modules
'''
            result = subprocess.run([sys.executable, '-c', script, str(output)], cwd=runner.HERE,
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Execution blocked', result.stderr)
            self.assertFalse(output.exists())

    def test_no_subset_or_force_cli(self):
        result = subprocess.run([sys.executable, str(runner.HERE / 'run_transfer.py'), '--force'],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('unrecognized arguments', result.stderr)

    def test_normalization_formula_and_bias(self):
        vectors = {'w_false': [1.] + [0.] * 63, 'w_true': [0.] * 64,
                   'ln_gamma': [1.] * 64, 'ln_beta': [2.] * 64}
        norm = runner.normalization(vectors, .5)
        self.assertEqual(norm['center'], 2.)
        self.assertAlmostEqual(norm['radius'], 63 ** .5)
        self.assertAlmostEqual(norm['raw_cutoff'], 2 + .5 * 63 ** .5)
        vectors['ln_gamma'] = [0.] * 64
        with self.assertRaises(ValueError):
            runner.normalization(vectors, .5)

    def test_selection_is_signed_strict_and_keeps_original_order(self):
        norm = {'center': 0., 'radius': 2., 'normalized_cutoff': .5}
        z, labels, selected = runner.select([1., -.5, 9., .2, .1], norm, 2)
        self.assertEqual(labels, ['rejected', 'accepted', 'rejected', 'accepted', 'accepted'])
        self.assertEqual(selected, {'accepted': [1, 3], 'rejected': [0, 2]})

    def test_partial_forward_cost_is_recorded_as_attempted(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger = runner.ForwardLedger(Path(temp) / 'events.jsonl')
            model = FakeModel()
            ledger.attach(model, 'float32')
            model.call(4)
            with self.assertRaises(RuntimeError):
                model.call(3, fail=True)
            cost = ledger.snapshot()
            self.assertEqual(cost['sequence_forwards_attempted'], 7)
            self.assertEqual(cost['sequence_forwards_completed'], 4)
            self.assertTrue(cost['failed_forward_internal_progress_unknown'])

    def make_context(self, directory):
        return {'directory': directory, 'status': {'task': {'model_id': 'fake', 'n_layer': 2,
                 'model_heads': 2, 'head': 1}, 'status': 'running'},
                'ledger': runner.ForwardLedger(directory / 'forward_events.jsonl'),
                'cohort_lock': {}, 'handles': [], 'sources': {'fake.py': 'abc'}}

    def payload(self):
        candidates = [{'candidate_id': str(i), 'candidate_index': i, 'recipient': '()'} for i in range(4)]
        templates = [{'family_id': 'template' + str(i), 'recipient': '()', 'donors': [
                     {'cell': name, 'position': p, 'string': '()'} for name, p in zip(runner.ANCHORS, (20, 28))]}
                     for i in range(4)]
        return candidates, templates

    def run_screen(self, context, values32, values64):
        runtimes = {'float32': FakeRuntime(values32), 'float64': FakeRuntime(values64)}
        plan = {'screen': {'families_per_stratum': 2}, 'intervention': {'numerical_allowance_nat': .001}}
        runner.screen_head(context, self.payload(), {'center': 0., 'radius': 1., 'normalized_cutoff': 1.},
                           plan, lambda task, lock, dtype: runtimes[dtype],
                           SimpleNamespace(float32='float32', float64='float64'),
                           lambda strings, result: [2] * len(strings))

    def test_screening_receipt_and_template_assignment_precede_anchors(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            context = self.make_context(directory)
            self.run_screen(context, [.2, 2., .3, 3.], [.2, 2., .3, 3.])
            self.assertEqual(context['status']['status'], 'screened')
            families = context['families']
            self.assertEqual([f['candidate_index'] for f in families], [0, 2, 1, 3])
            self.assertEqual([f['donor_template_id'] for f in families], ['template0', 'template1', 'template2', 'template3'])
            self.assertTrue((directory / 'screening_receipt.json').exists())
            self.assertFalse((directory / 'cases.jsonl').exists())
            self.assertEqual(context['ledger'].snapshot()['sequence_forwards_completed'], 8)

    def test_shortfall_has_no_receipt_or_anchors(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            context = self.make_context(directory)
            self.run_screen(context, [.2, .3, .4, 3.], [.2, .3, .4, 3.])
            self.assertEqual(context['status']['status'], 'insufficient_yield')
            self.assertTrue((directory / 'screening.jsonl').exists())
            self.assertFalse((directory / 'screening_receipt.json').exists())

    def test_any_candidate_dtype_disagreement_fails_whole_head(self):
        with tempfile.TemporaryDirectory() as temp:
            context = self.make_context(Path(temp))
            with self.assertRaises(AssertionError):
                self.run_screen(context, [.2, 2., .9999, 3.], [.2, 2., 1.0001, 3.])
            self.assertFalse((Path(temp) / 'screening_receipt.json').exists())

    def test_anchor_gate_uses_signed_difference_not_separate_margin_error(self):
        with tempfile.TemporaryDirectory() as temp:
            context = self.make_context(Path(temp))
            self.run_screen(context, [.2, 2., .3, 3.], [.2, 2., .3, 3.])
            contexts = list(context['runtimes'].values())
            def fake_transfer(runtime, recipients, donors, recipient_positions, donor_positions):
                values = [0. if p == 20 else .5 for p in donor_positions]
                if runtime is contexts[0]:
                    values = [v + (.0008 if p == 20 else -.0008) for v, p in zip(values, donor_positions)]
                return SimpleNamespace(controls={'fake': True}, snapshots=lambda: [{'patched_margin': v} for v in values])
            with self.assertRaises(AssertionError):
                runner.anchor_head(context, {'intervention': {'strict_gap_nat': .202, 'numerical_allowance_nat': .001}},
                                   fake_transfer)
            self.assertAlmostEqual(context['controls']['anchor_precision']['maximum_signed_contrast_dtype_difference'], .0016)
            self.assertTrue((Path(temp) / 'cases.jsonl').exists())

    def test_boundary_straddle_fails_even_when_dtype_difference_is_small(self):
        with tempfile.TemporaryDirectory() as temp:
            context = self.make_context(Path(temp))
            self.run_screen(context, [.2, 2., .3, 3.], [.2, 2., .3, 3.])
            runtime32 = context['runtimes']['float32']
            def fake_transfer(runtime, recipients, donors, recipient_positions, donor_positions):
                gap = .2021 if runtime is runtime32 else .2019
                return SimpleNamespace(controls={}, snapshots=lambda: [
                    {'patched_margin': 0. if p == 20 else gap} for p in donor_positions])
            with self.assertRaises(AssertionError):
                runner.anchor_head(context, {'intervention': {'strict_gap_nat': .202, 'numerical_allowance_nat': .001}},
                                   fake_transfer)
            self.assertEqual(context['controls']['anchor_precision']['separation_boundary_straddles'], 4)


if __name__ == '__main__':
    unittest.main()
