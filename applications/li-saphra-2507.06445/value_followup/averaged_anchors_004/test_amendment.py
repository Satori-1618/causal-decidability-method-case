"""Amendment-only fake orchestration checks; original tests remain unchanged."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import analysis
import descriptives
import run as runner
import test_run


class AmendmentTests(unittest.TestCase):
    def fake(self, directory, separating):
        helper = test_run.RunnerTests(methodName='runTest')
        helper.setUp()
        return helper.run_fake(directory, separating=separating)

    def test_descriptive_receipt_precedes_targets_and_primary_summary_unchanged(self):
        stage = runner.transfer_stage
        before_targets = []
        def check(context, role, transfer):
            if role == 'target':
                path = context['directory']
                self.assertFalse((path / 'cases.jsonl').exists())
                receipt = json.loads((path / 'forecast_receipt.json').read_text())
                self.assertEqual(receipt['descriptive_forecasts_sha256'], runner.sha(path / 'descriptive_forecasts.jsonl'))
                rows = [json.loads(x) for x in (path / 'descriptive_forecasts.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows), 256)
                for row, record in zip(rows, context['records']):
                    self.assertEqual(row['family_id'], record['family_id'])
                    self.assertEqual(row['forecasts'], descriptives.forecasts(runner.role_margins(record, 'calibration')))
                    self.assertEqual(len(record['cells']), 8)
                before_targets.append(True)
            return stage(context, role, transfer)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(runner, 'transfer_stage', side_effect=check):
            path = Path(tmp) / 'run'
            manifest, _ = self.fake(path, 128)
            self.assertEqual(before_targets, [True])
            self.assertEqual(manifest['status'], 'completed')
            cases = [json.loads(x) for x in (path / 'cases.jsonl').read_text().splitlines()]
            self.assertEqual(json.loads((path / 'analysis_summary.json').read_text()),
                             analysis.cohort_result([row['analysis'] for row in cases]))
            summary = json.loads((path / 'descriptive_summary.json').read_text())
            self.assertEqual(summary['all_families']['counts']['families'], 256)
            self.assertEqual(summary['separating_subset']['counts']['families'], 128)
            for name in ('descriptive_forecasts.jsonl', 'descriptive_cases.jsonl', 'descriptive_summary.json'):
                self.assertEqual(manifest['output_hashes'][name], runner.sha(path / name))

    def test_failed_original_start_saves_forecasts_without_descriptive_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'run'
            manifest, calls = self.fake(path, 127)
            self.assertEqual(manifest['status'], 'insufficient_design_yield')
            self.assertTrue((path / 'descriptive_forecasts.jsonl').exists())
            self.assertFalse((path / 'descriptive_cases.jsonl').exists())
            self.assertFalse((path / 'descriptive_summary.json').exists())
            self.assertEqual(calls, [('calibration', 2048), ('calibration', 2048)])


if __name__ == '__main__':
    unittest.main()
