"""Source-only and fake-forward integration tests for the history adapter."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import run as runner


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
    def register_forward_pre_hook(self, hook):
        self.before.append(hook)
        return SimpleNamespace(remove=lambda: self.before.remove(hook))
    def register_forward_hook(self, hook):
        self.after.append(hook)
        return SimpleNamespace(remove=lambda: self.after.remove(hook))
    def call(self, rows, fail=False):
        args = (SimpleNamespace(shape=(rows, 42)),)
        for hook in self.before:
            hook(self, args)
        if fail:
            raise RuntimeError('Synthetic forward failure')
        for hook in self.after:
            hook(self, args, None)


class FakeRuntime:
    def __init__(self, margin):
        self.margin = margin
        self.model = FakeModel()
    def run(self, strings):
        self.model.call(len(strings))
        attention = [[0.] * 42 for _ in strings]
        for row in attention:
            row[17] = .5
        return SimpleNamespace(margins=Array([self.margin] * len(strings)), attention=Array(attention))


def payload():
    candidates = [{'candidate_id': str(i), 'candidate_index': i, 'recipient': '(' * 16 + ')' * 16}
                  for i in range(2048)]
    # These symbolic strings are fake-transfer inputs, never neural inputs.
    donors = [{'role': role, 'replica': replica, 'recency': recency, 'ending': suffix, 'position': 28,
               'cell': f'{role}_t{recency}_{suffix}_{replica}',
               'string': f'{role}_t{recency}_{suffix}_{replica}'}
              for role in ('calibration', 'target') for replica in (0, 1)
              for recency in (6, 10) for suffix in ('alt', 'close')]
    families = [{'family_id': 'template' + str(i), 'input_hash_partition': 'confirmation',
                 'donors': copy.deepcopy(donors)} for i in range(256)]
    return candidates, families


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((runner.HERE / 'plan.json').read_text())

    def test_import_is_model_free_and_does_not_repoint_frozen_module(self):
        script = "import run,sys;assert 'torch' not in sys.modules;assert run.shared.HERE.name=='averaged_anchors_004'"
        result = subprocess.run([sys.executable, '-c', script], cwd=runner.HERE, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_pending_release_always_stops_before_model_or_output(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'forbidden'
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

    def test_scientific_plan_comparison_detects_deletion(self):
        modified = copy.deepcopy(self.plan)
        modified['status'] = 'released_for_execution'
        modified['execution_authorized'] = True
        self.assertEqual(runner.scientific_plan(self.plan), runner.scientific_plan(modified))
        del modified['inference']['absolute_adequacy_test']
        self.assertNotEqual(runner.scientific_plan(self.plan), runner.scientific_plan(modified))

    def run_fake(self, output, separating=64, margin=0., fail=False):
        analysis = runner.load_module('history_005_test_analysis', runner.HERE / 'analysis.py')
        ledger_path = runner.VALUE / 'screen_transfer_003'
        sys.path.insert(0, str(ledger_path))
        ledger = runner.load_module('history_005_test_ledger', ledger_path / 'run_transfer.py').ForwardLedger
        calls = []
        def load_runtime(task, lock, dtype):
            self.assertEqual(task, {'model_id': 'a9g0io1r', 'n_layer': 2, 'n_head': 4, 'head': 1})
            return FakeRuntime(margin)
        def transfer(runtime, recipients, donors, recipient_positions, donor_positions):
            role = 'calibration' if len(calls) < 2 else 'target'
            calls.append((role, len(donors)))
            self.assertEqual(set(donor_positions), {28})
            if fail:
                runtime.model.call(64, fail=True)
            if role == 'target':
                receipt = json.loads((output / 'forecast_receipt.json').read_text())
                self.assertTrue(receipt['start_rule']['start_targets'])
                self.assertEqual(receipt['forecasts_sha256'], runner.shared.sha(output / 'forecasts.jsonl'))
            for _ in range(4):
                runtime.model.call(len(donors))
            values = [(1. + (-.04 if '_t6_' in donor else .04)) if i // 8 < separating else 1.
                      for i, donor in enumerate(donors)]
            return SimpleNamespace(controls={'synthetic': True}, snapshots=lambda: [{'patched_margin': v} for v in values])
        torch = SimpleNamespace(float32='float32', float64='float64', __version__='fake',
                                set_num_threads=lambda n: None, use_deterministic_algorithms=lambda b: None)
        dependencies = (torch, load_runtime, transfer, lambda strings, native: [17] * len(strings), ledger, analysis)
        candidates, families = payload()
        with mock.patch.object(runner, 'execution_dependencies', return_value=dependencies):
            runner.execute(output, self.plan, {'final_review': {'reviewed_commit': '0' * 40}}, {}, {},
                           {'fake': True}, candidates, families)
        return json.loads((output / 'manifest.json').read_text()), calls, analysis

    def test_63_stops_after_calibration_without_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'run'
            manifest, calls, _ = self.run_fake(output, separating=63)
            self.assertEqual(manifest['status'], 'insufficient_design_yield')
            self.assertEqual(manifest['separating_families'], 63)
            self.assertEqual(calls, [('calibration', 2048), ('calibration', 2048)])
            self.assertEqual(manifest['cost']['sequence_forwards_completed'], 20480)
            self.assertTrue((output / 'forecast_receipt.json').exists())
            self.assertFalse((output / 'cases.jsonl').exists())

    def test_64_starts_all256_targets_with_all_four_frozen_forecasts(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'run'
            manifest, calls, analysis = self.run_fake(output)
            self.assertEqual(manifest['status'], 'completed')
            self.assertEqual(calls, [('calibration', 2048), ('calibration', 2048), ('target', 2048), ('target', 2048)])
            self.assertEqual(manifest['cost']['sequence_forwards_completed'], 36864)
            records = [json.loads(line) for line in (output / 'cases.jsonl').read_text().splitlines()]
            self.assertEqual(len(records), 256)
            self.assertTrue(all(len(row['cells']) == 16 for row in records))
            self.assertEqual(sum(row['separating'] for row in records), 64)
            for row in records:
                self.assertEqual(set(row['forecasts']['float64']), {'H_recency', 'H_suffix', 'H_constant', 'H_cell'})
                self.assertEqual(row['forecasts'], row['analysis']['forecasts'])
            summary = json.loads((output / 'analysis_summary.json').read_text())
            self.assertEqual(summary, analysis.cohort_result([row['analysis'] for row in records]))
            receipt = json.loads((output / 'forecast_receipt.json').read_text())
            self.assertLessEqual(receipt['written_before_targets_at'], manifest['target_measurement_started_at'])
            self.assertEqual(manifest['output_hashes']['analysis_summary.json'], runner.shared.sha(output / 'analysis_summary.json'))

    def test_native_shortfall_has_no_calibration(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'run'
            manifest, calls, _ = self.run_fake(output, margin=8.)
            self.assertEqual(manifest['status'], 'insufficient_screening_yield')
            self.assertEqual(calls, [])
            self.assertEqual(manifest['cost']['sequence_forwards_completed'], 4096)

    def test_failure_preserves_cost_and_never_retries(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'run'
            with self.assertRaisesRegex(RuntimeError, 'Synthetic forward failure'):
                self.run_fake(output, fail=True)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'technical_failure')
            self.assertEqual(manifest['cost']['sequence_forwards_attempted'], 4160)
            self.assertEqual(manifest['cost']['sequence_forwards_completed'], 4096)
            self.assertTrue(manifest['cost']['failed_forward_internal_progress_unknown'])
            self.assertFalse((output / 'cases.jsonl').exists())

    def test_no_force_or_subset_override(self):
        result = subprocess.run([sys.executable, str(runner.HERE / 'run.py'), '--force'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('unrecognized arguments', result.stderr)


if __name__ == '__main__':
    unittest.main()
