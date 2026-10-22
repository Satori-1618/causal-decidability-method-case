"""Fake-only orchestration tests. No torch import or neural model construction."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import analysis
import run as runner


class Array:
    def __init__(self, data):
        self.data = data
    def tolist(self):
        return self.data
    def __getitem__(self, index):
        return Array(self.data[index])


class FakeModel:
    def __init__(self):
        self.before, self.after = [], []
    def register_forward_pre_hook(self, hook):
        self.before.append(hook)
        return SimpleNamespace(remove=lambda: self.before.remove(hook))
    def register_forward_hook(self, hook):
        self.after.append(hook)
        return SimpleNamespace(remove=lambda: self.after.remove(hook))
    def call(self, size, fail=False):
        args = (SimpleNamespace(shape=(size, 42)),)
        for hook in self.before:
            hook(self, args)
        if fail:
            raise RuntimeError('Synthetic failed forward')
        for hook in self.after:
            hook(self, args, None)


class FakeRuntime:
    def __init__(self, margin):
        self.model = FakeModel()
        self.margin = margin
    def run(self, strings):
        self.model.call(len(strings))
        rows = [[0.] * 42 for _ in strings]
        for row in rows:
            row[17] = .4
        return SimpleNamespace(margins=Array([self.margin] * len(strings)), attention=Array(rows))


def ledger_class():
    directory = runner.VALUE / 'screen_transfer_003'
    sys.path.insert(0, str(directory))
    return runner.load_module('averaged_test_previous_runner', directory / 'run_transfer.py').ForwardLedger


def payload():
    candidates = [{'candidate_id': 'case' + str(i), 'candidate_index': i, 'recipient': '(' * 16 + ')' * 16}
                  for i in range(2048)]
    donors = [{'role': role, 'cell': '{}_{}_{}_{}'.format(role, sign, p, r), 'balance': d,
               'position': p, 'replica': r,
               'string': '(' * 16 + ')' * 16 if d < 0 else ')' * 16 + '(' * 16}
              for role in ('calibration', 'target') for sign, d in (('neg', -2), ('pos', 2))
              for p in (20, 28) for r in (0, 1)]
    families = [{'family_id': 'template' + str(i), 'input_hash_partition': 'confirmation',
                 'donors': copy.deepcopy(donors)} for i in range(256)]
    return candidates, families


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((runner.HERE / 'plan.json').read_text())
        self.release = {'final_review': {'reviewed_commit': '0' * 40}}

    def test_import_never_loads_torch(self):
        result = subprocess.run([sys.executable, '-c', 'import run,sys;assert "torch" not in sys.modules'],
                                cwd=runner.HERE, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_pending_fixture_blocks_before_runtime_or_output_even_after_future_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'forbidden'
            script = '''
import json,sys
from pathlib import Path
import run as r
original=Path.read_text
def read(path,*args,**kwargs):
    text=original(path,*args,**kwargs)
    if path==r.HERE/'plan.json':
        p=json.loads(text);p['status']='awaiting_final_pre_run_review';p['execution_authorized']=False
        return json.dumps(p)
    return text
Path.read_text=read
sys.argv=['run.py','--execute','--output',sys.argv[1]]
try:r.main()
finally:assert 'torch' not in sys.modules
'''
            result = subprocess.run([sys.executable, '-c', script, str(output)], cwd=runner.HERE,
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Execution blocked', result.stderr)
            self.assertFalse(output.exists())

    def test_screen_uses_signed_strict_margin_and_first_order(self):
        self.assertEqual(runner.selected_indices([8., -9., 7.99, 0.], 8., 2), [1, 2])

    def run_fake(self, output, separating=128, margin=2., fail_calibration=False):
        calls = []
        def load_runtime(task, lock, dtype):
            self.assertEqual(task, {'model_id': 'a9g0io1r', 'n_layer': 2, 'n_head': 4, 'head': 1})
            return FakeRuntime(margin)
        def transfer(runtime, recipients, donors, positions, donor_positions):
            stage = 'calibration' if len(calls) < 2 else 'target'
            calls.append((stage, len(recipients)))
            if fail_calibration:
                runtime.model.call(64, fail=True)
            if stage == 'target':
                receipt = json.loads((output / 'forecast_receipt.json').read_text())
                self.assertTrue(receipt['start_rule']['start_targets'])
                self.assertEqual(receipt['forecasts_sha256'], runner.sha(output / 'forecasts.jsonl'))
            for _ in range(4):
                runtime.model.call(len(recipients))
            values = [(-.5 if donor[0] == '(' else .5) if i // 8 < separating else 0.
                      for i, donor in enumerate(donors)]
            return SimpleNamespace(controls={'fake_operator': True}, snapshots=lambda: [{'patched_margin': x} for x in values])
        fake_torch = SimpleNamespace(float32='float32', float64='float64', __version__='synthetic',
                                     set_num_threads=lambda n: None, use_deterministic_algorithms=lambda b: None)
        dependencies = (fake_torch, load_runtime, transfer, lambda strings, native: [17] * len(strings),
                        ledger_class(), analysis)
        candidates, families = payload()
        with mock.patch.object(runner, 'execution_dependencies', return_value=dependencies), \
                mock.patch.object(runner.subprocess, 'check_output', return_value='a'*40+'\n'):
            runner.execute(output, self.plan, self.release, {}, {}, {'synthetic': True}, candidates, families)
        return json.loads((output / 'manifest.json').read_text()), calls

    def test_127_separating_stops_after_all_calibration_and_saves_forecasts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'run'
            manifest, calls = self.run_fake(output, separating=127)
            self.assertEqual(manifest['status'], 'insufficient_design_yield')
            self.assertEqual(manifest['separating_families'], 127)
            self.assertEqual(calls, [('calibration', 2048), ('calibration', 2048)])
            self.assertEqual(manifest['cost']['sequence_forwards_completed'], 20480)
            self.assertTrue((output / 'forecast_receipt.json').exists())
            self.assertFalse((output / 'cases.jsonl').exists())
            self.assertNotIn('target_measurement_started_at', manifest)

    def test_128_starts_all256_targets_and_receipt_precedes_every_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'run'
            manifest, calls = self.run_fake(output, separating=128)
            self.assertEqual(manifest['status'], 'completed')
            self.assertEqual(calls, [('calibration', 2048), ('calibration', 2048), ('target', 2048), ('target', 2048)])
            self.assertEqual(manifest['cost']['sequence_forwards_completed'], 36864)
            rows = [json.loads(line) for line in (output / 'cases.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 256)
            self.assertTrue(all(len(row['cells']) == 16 for row in rows))
            self.assertEqual(sum(row['separating'] for row in rows), 128)
            receipt = json.loads((output / 'forecast_receipt.json').read_text())
            self.assertLessEqual(receipt['written_before_targets_at'], manifest['target_measurement_started_at'])
            self.assertEqual(manifest['cost']['target_jobs_completed_by_dtype'], {'float32': 2048, 'float64': 2048})
            summary = json.loads((output / 'analysis_summary.json').read_text())
            self.assertEqual(summary, analysis.cohort_result([row['analysis'] for row in rows]))
            self.assertEqual(summary['paired_averaging_on_all_families']['families'], 256)
            self.assertFalse(summary['future_confirmation_permission'])
            self.assertEqual(manifest['output_hashes']['analysis_summary.json'], runner.sha(output / 'analysis_summary.json'))

    def test_within_cell_straddle_is_reported_instead_of_hardcoded_zero(self):
        calibration = {name: (-.5 if '_neg_' in name else .5) for name in analysis.CALIBRATION_CELLS}
        predictions = analysis.forecasts(calibration)
        row = {'family_id': 'synthetic', 'cells': {},
               'forecasts': {tag: copy.deepcopy(predictions) for tag in runner.DTYPES}, 'separating': True}
        for name, value in calibration.items():
            row['cells'][name] = {'donor_metadata': {'role': 'calibration'},
                                  **{tag: {'patched_margin': value} for tag in runner.DTYPES}}
        for name, value in predictions['B_avg'].items():
            row['cells'][name] = {'donor_metadata': {'role': 'target'},
                                  **{tag: {'patched_margin': value} for tag in runner.DTYPES}}
        row['cells']['target_neg_20_1']['float32']['patched_margin'] += .2021
        row['cells']['target_neg_20_1']['float64']['patched_margin'] += .2019
        with tempfile.TemporaryDirectory() as tmp:
            context = {'directory': Path(tmp), 'records': [row], 'controls': {}}
            runner.target_precision(context, analysis)
            control = context['controls']['target_precision']
            self.assertEqual(control['same_cell_boundary_straddles'], 1)
            self.assertEqual(control['same_cell_boundary_straddle_families'], 1)
            self.assertEqual(row['analysis']['within_cell_numerically_unresolved'], ['neg_20'])
            self.assertFalse(row['analysis']['extra_prefix_dependence_witness'])
            self.assertTrue(row['analysis']['extra_prefix_dependence_possible_witness'])

    def test_native_shortfall_stops_without_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'run'
            manifest, calls = self.run_fake(output, margin=8.)
            self.assertEqual(manifest['status'], 'insufficient_screening_yield')
            self.assertEqual(calls, [])
            self.assertEqual(manifest['cost']['sequence_forwards_completed'], 4096)
            self.assertFalse((output / 'calibration_cases.jsonl').exists())

    def test_failed_forward_records_partial_cost_and_never_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'run'
            with self.assertRaisesRegex(RuntimeError, 'Synthetic failed forward'):
                self.run_fake(output, fail_calibration=True)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'technical_failure')
            self.assertEqual(manifest['cost']['sequence_forwards_attempted'], 4160)
            self.assertEqual(manifest['cost']['sequence_forwards_completed'], 4096)
            self.assertTrue(manifest['cost']['failed_forward_internal_progress_unknown'])
            self.assertFalse((output / 'cases.jsonl').exists())

    def test_no_force_or_subset_cli(self):
        result = subprocess.run([sys.executable, str(runner.HERE / 'run.py'), '--force'],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('unrecognized arguments', result.stderr)


if __name__ == '__main__':
    unittest.main()
