import importlib.util
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import causal_decidability as cd  # noqa: E402
from causal_decidability import calculator  # noqa: E402

spec = importlib.util.spec_from_file_location('twelve_cell', ROOT / 'examples' / 'twelve_cell.py')
twelve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(twelve)


class Calculator(unittest.TestCase):

    def test_half_ulp_is_half_the_machine_epsilon_for_every_dtype(self):
        # IEEE / bfloat16 machine epsilon = 2**-(stored mantissa bits)
        for name, eps in (('bfloat16', 2.0 ** -7), ('float16', 2.0 ** -10),
                          ('float32', 2.0 ** -23), ('float64', 2.0 ** -52)):
            self.assertEqual(calculator.half_ulp(name), eps / 2)
        try:
            import torch
        except ImportError:
            return
        for name in ('bfloat16', 'float16', 'float32', 'float64'):
            self.assertEqual(calculator.half_ulp(name), torch.finfo(getattr(torch, name)).eps / 2)

    def test_zero_separation_is_never_decidable(self):
        out = cd.decidability({'a': 1.0, 'b': 1.0}, n=10**6, sigma=1e-9, dtype='float64',
                              readout_scale=1.0)
        self.assertEqual(out['separation'], 0.0)
        self.assertFalse(out['decidable'])
        self.assertEqual(out['lever']['kind'], 'intervention_or_candidates')

    def test_declared_equivalents_do_not_set_the_separation(self):
        out = cd.decidability({'a': 1.0, 'a2': 1.001, 'b': 0.0}, n=100, sigma=0.1,
                              dtype='float32', readout_scale=1.0,
                              equivalence_groups=[('a', 'a2')])
        self.assertAlmostEqual(out['separation'], 1.0)

    def test_statistical_floor_and_ratio(self):
        z = cd.bonferroni_z(0.01, 2)
        self.assertAlmostEqual(z, 2.8070, places=4)
        out = cd.decidability({'inert': 0.0, 'all': 4.0}, n=100, sigma=2.0, dtype='float32',
                              readout_scale=1.0, noise_factor=1.0, signatures=2)
        self.assertAlmostEqual(out['statistical_floor'], z * 2.0 / 10.0)
        self.assertAlmostEqual(out['ratio'], 4.0 / (z * 0.2))
        self.assertEqual(out['binding_floor'], 'statistical')

    def test_lever_names_the_sample_size_that_would_reach_ratio_one(self):
        out = cd.decidability({'a': 0.0, 'b': 0.1}, n=10, sigma=1.0, dtype='float32',
                              readout_scale=1.0)
        needed = out['lever']['replicates_needed']
        again = cd.decidability({'a': 0.0, 'b': 0.1}, n=needed, sigma=1.0, dtype='float32',
                                readout_scale=1.0)
        self.assertGreaterEqual(again['ratio'], 0.999)

    def test_non_finite_declarations_are_refused_not_judged(self):
        base = dict(predictions={'a': 0.0, 'b': 1.0}, n=10, sigma=1.0, dtype='float32',
                    readout_scale=1.0)
        for bad in (dict(sigma=float('nan')), dict(sigma=float('inf')), dict(n=float('nan')),
                    dict(readout_scale=float('nan')), dict(z=float('nan')),
                    dict(predictions={'a': 0.0, 'b': float('nan')}),
                    dict(predictions={'a': 0.0, 'b': float('inf')})):
            with self.assertRaises(ValueError, msg=str(bad)):
                cd.decidability(**dict(base, **bad))

    def test_numerical_floor_binds_in_low_precision(self):
        out = cd.decidability({'a': 0.0, 'b': 1.0}, n=10**6, sigma=0.01, dtype='bfloat16',
                              readout_scale=10.0, depth=768)
        self.assertEqual(out['binding_floor'], 'numerical')
        self.assertEqual(out['lever']['kind'], 'precision')


class CompatibleSet(unittest.TestCase):

    def test_four_outcomes(self):
        pred = {'a': 0.0, 'b': 1.0, 'c': 2.0}
        self.assertEqual(cd.compatible_set(pred, 0.05, 0.1)['outcome'], 'resolved')
        self.assertEqual(cd.compatible_set(pred, 0.5, 0.6)['outcome'], 'partially_resolved')
        self.assertEqual(cd.compatible_set(pred, 5.0, 0.1)['outcome'], 'no_candidate_fits')
        self.assertEqual(cd.compatible_set(pred, 1.0, 5.0)['outcome'],
                         'insufficient_evidence')

    def test_a_declared_group_is_never_split(self):
        pred = {'twin': 1.0, 'twin2': 1.005, 'other': 0.0}
        radius = {'twin': 0.004, 'twin2': 0.0041, 'other': 0.004}
        out = cd.compatible_set(pred, 1.0001, radius, equivalence_groups=[('twin', 'twin2')])
        self.assertEqual(out['retained'], ['twin', 'twin2'])
        self.assertEqual(out['outcome'], 'resolved')
        split = cd.compatible_set(pred, 1.0001, radius)
        self.assertEqual(split['retained'], ['twin'])

    def test_a_failed_measurement_is_an_error_not_an_exclusion(self):
        for bad in (float('nan'), float('inf'), float('-inf')):
            with self.assertRaises(ValueError):
                cd.compatible_set({'inert': 0.0, 'all': 1.0}, bad, 0.1)
            with self.assertRaises(ValueError):
                cd.compatible_set({'inert': bad, 'all': 1.0}, 0.0, 0.1)
            with self.assertRaises(ValueError):
                cd.compatible_set({'inert': 0.0, 'all': 1.0}, 0.0, bad)
        with self.assertRaises(ValueError):
            cd.compatible_set({'a': (0.0, 1.0)}, (0.0, float('nan')), 0.1)

    def test_every_cell_must_be_compatible(self):
        pred = {'a': (0.0, 1.0), 'b': (0.0, 0.0)}
        out = cd.compatible_set(pred, (0.0, 0.9), (0.2, 0.2))
        self.assertEqual(out['retained'], ['a'])

    def test_bad_declarations_are_refused(self):
        with self.assertRaises(ValueError):
            cd.compatible_set({'a': 0.0}, 0.0, 0.1, equivalence_groups=[('a', 'x')])
        with self.assertRaises(ValueError):
            cd.compatible_set({'a': (0.0, 1.0)}, 0.0, 0.1)
        with self.assertRaises(ValueError):
            cd.compatible_set({'a': 0.0, 'b': 1.0, 'c': 2.0}, 0.0, 0.1,
                              equivalence_groups=[('a', 'b'), ('b', 'c')])


class DesignInputs(unittest.TestCase):
    """Regressions from the method-readiness review of 2026-09-21."""

    def test_rows_of_unequal_length_are_refused_not_truncated(self):
        with self.assertRaises(ValueError):
            cd.signatures({'a': (0.0,), 'b': (0.0, 100.0)})
        with self.assertRaises(ValueError):
            cd.separating_cells((0.0,), (0.0, 100.0))

    def test_non_finite_predictions_are_refused(self):
        with self.assertRaises(ValueError):
            cd.signatures({'a': (float('nan'),), 'b': (float('nan'),)})
        with self.assertRaises(ValueError):
            cd.separating_cells((float('nan'),), (float('nan'),))
        with self.assertRaises(ValueError):
            cd.signatures({})

    def test_exact_equivalence_and_tolerance_summary_are_different_things(self):
        pred = {'a': (0.0,), 'b': (0.09,), 'c': (0.18,)}
        self.assertEqual(cd.signatures(pred), [['a'], ['b'], ['c']])
        self.assertEqual(cd.groups_within(pred, 0.1), [['a', 'b', 'c']])
        self.assertEqual(cd.separating_cells(pred['a'], pred['c'], 0.1), [0])

    def test_exact_mcnemar_stays_finite_for_many_discordant_units(self):
        self.assertEqual(cd.exact_mcnemar_p(512, 512), 1.0)
        p = cd.exact_mcnemar_p(400, 600)
        self.assertTrue(0.0 < p < 1e-9)
        self.assertEqual(cd.exact_mcnemar_p(7, 0), 0.015625)

    def test_empty_candidate_set_is_a_declaration_error(self):
        with self.assertRaises(ValueError):
            cd.compatible_set({}, 0.0, float('nan'))
        with self.assertRaises(ValueError):
            cd.compatible_set({}, 0.0, 0.1)
        with self.assertRaises(ValueError):
            cd.compatible_set({'a': 0.0, 'b': 1.0}, 0.0, {'a': 0.1})


class Paired(unittest.TestCase):

    def test_reproduces_the_published_makelov_accuracy_contrast_from_its_counts(self):
        # rowspace vs nullspace interchange, n = 2000: 6 both, 7 rowspace only, 0 nullspace
        # only, 1987 neither (applications/makelov-2311.17030, paired_resolution.json)
        a = [1] * 6 + [1] * 7 + [0] * 1987
        b = [1] * 6 + [0] * 7 + [0] * 1987
        out = cd.paired(a, b)
        self.assertEqual(out['readout'], 'binary')
        self.assertAlmostEqual(out['separation'], 0.0035)
        self.assertAlmostEqual(out['paired_ratio'], 2.650, places=3)
        self.assertAlmostEqual(out['unpaired_ratio'], 1.610, places=3)
        self.assertEqual(out['exact_mcnemar_p'], 0.015625)

    def test_pairing_resolves_what_the_means_cannot(self):
        base = [((i * 7919) % 1000) / 100.0 for i in range(400)]
        a = [x + 0.3 + ((i * 31) % 7 - 3) * 0.01 for i, x in enumerate(base)]
        out = cd.paired(a, base)
        self.assertEqual(out['readout'], 'continuous')
        self.assertGreater(out['paired_ratio'], 10 * out['unpaired_ratio'])

    def test_bad_inputs_are_refused(self):
        for a, b in (([1, 0, 1], [1, 0]), ([1.0], [0.0]), ([0.5, float('nan')], [0.1, 0.2]),
                     ([0.5, float('inf')], [0.1, 0.2])):
            with self.assertRaises(ValueError):
                cd.paired(a, b)
        self.assertTrue(math.isnan(cd.paired([0, 1], [0, 1])['exact_mcnemar_p']))


class FromData(unittest.TestCase):
    """The data-to-decision recipe under an explicit contract (review of 2026-09-21)."""

    CANDS = {'A': {'read_row': 'same_as:full', 'read_null': 'same_as:baseline'},
             'B': {'read_row': 'same_as:baseline', 'read_null': 'same_as:full'}}
    TESTED = ['read_row', 'read_null']

    def setUp(self):
        import csv
        from causal_decidability import evaluate as ev
        self.ev = ev
        with open(ROOT / 'examples' / 'data' / 'makelov_read_source.csv', newline='') as h:
            self.data = ev.load_rows((r['unit'], r['repeat'], r['condition'], r['value'])
                                     for r in csv.DictReader(h))
        self.candidates = json.loads(
            (ROOT / 'examples' / 'data' / 'makelov_read_source_candidates.json').read_text())

    def _units(self, per_unit, n=12):
        rows = []
        for unit in range(n):
            for repeat, values in enumerate(per_unit(unit)):
                rows += [(unit, repeat, c, v) for c, v in values.items()]
        return self.ev.load_rows(rows)

    def test_the_pilot_loss_is_reproduced_exactly(self):
        self.assertEqual(len(self.data), 32)
        out = self.ev.evaluate(self.data, self.candidates, self.TESTED, 'absolute',
                               tolerance_fraction=0.25, scale=('full', 'baseline'),
                               scope='pooled')
        loss = {r['candidate']: r['loss'] for r in out['rows']}
        self.assertAlmostEqual(loss['A_visible_read'], 1.338178314268589, places=9)
        self.assertAlmostEqual(loss['B_null_read'], 0.260309673845768, places=9)
        comparison = out['comparisons'][0]
        self.assertAlmostEqual(comparison['mean_difference'], 1.077868640422821, places=9)
        self.assertEqual(comparison['units_favouring'], {'A_visible_read': 0, 'B_null_read': 32})
        self.assertEqual(out['status'], {'A_visible_read': 'excluded', 'B_null_read': 'adequate'})

    def test_absolute_errors_do_not_cancel_but_signed_ones_do(self):
        # per unit two directions (row, null) = (2, 3) and (-2, -1); baseline 0, full 1
        data = self._units(lambda u: [
            {'baseline': 0.0, 'full': 1.0, 'read_row': 2.0, 'read_null': 3.0},
            {'baseline': 0.0, 'full': 1.0, 'read_row': -2.0, 'read_null': -1.0}])
        absolute = self.ev.evaluate(data, self.CANDS, self.TESTED, 'absolute', tolerance=0.5,
                                    resamples=500)
        self.assertEqual(absolute['status']['B'], 'excluded')
        signed = self.ev.evaluate(data, self.CANDS, self.TESTED, 'signed', tolerance=0.0,
                                  resamples=500)
        self.assertNotEqual(signed['status']['B'], 'excluded')

    def test_a_relative_tolerance_carries_its_own_uncertainty(self):
        # residual always 2.5; reference gap 10 in 12 of 32 units, else 0; fraction 0.5.
        # Treating 0.5 * mean gap = 1.875 as fixed would exclude; jointly it is undecided.
        data = self._units(lambda u: [{'baseline': 0.0, 'full': 10.0 if u < 12 else 0.0,
                                       'probe': 2.5}], n=32)
        cands = {'zero': {'probe': 0.0}}
        out = self.ev.evaluate(data, cands, ['probe'], 'absolute', tolerance_fraction=0.5,
                               scale=('full', 'baseline'))
        self.assertEqual(out['status']['zero'], 'undecided')
        fixed = self.ev.evaluate(data, cands, ['probe'], 'absolute', tolerance=0.5 * 120 / 32)
        self.assertEqual(fixed['status']['zero'], 'excluded')

    def test_three_statuses_are_kept_apart(self):
        # probe values 0.2, 0.3, 0.4 (mean 0.3); 'edge' has loss 0.3, exactly the tolerance
        data = self._units(lambda u: [{'baseline': 0.0, 'full': 1.0,
                                       'probe': 0.2 + 0.1 * (u % 3)}], n=30)
        cands = {'near': {'probe': 0.3}, 'far': {'probe': 3.0}, 'edge': {'probe': 0.0}}
        out = self.ev.evaluate(data, cands, ['probe'], 'absolute', tolerance=0.3)
        self.assertEqual(out['status'], {'near': 'adequate', 'far': 'excluded',
                                         'edge': 'undecided'})
        self.assertEqual(out['outcome'], 'partially_resolved')

    def test_contract_and_inputs_are_checked(self):
        with self.assertRaises(ValueError):
            self.ev.evaluate(self.data, self.candidates, self.TESTED, None, tolerance=0.1)
        with self.assertRaises(ValueError):
            self.ev.evaluate(self.data, self.candidates, self.TESTED, 'absolute')
        with self.assertRaises(ValueError):
            self.ev.evaluate(self.data, self.candidates, self.TESTED, 'absolute',
                             tolerance=0.1, tolerance_fraction=0.1, scale=('full', 'baseline'))
        with self.assertRaises(ValueError):
            self.ev.evaluate(self._units(lambda u: [{'baseline': 0.0, 'full': 1.0,
                                                     'read_row': 0.0, 'read_null': 1.0}], n=5),
                             self.CANDS, self.TESTED, 'absolute', tolerance=0.1)
        with self.assertRaises(ValueError):
            self.ev.load_rows([('u', 0, 'c', 1.0), ('u', 0, 'c', 2.0)])
        with self.assertRaises(ValueError):
            self.ev.load_rows([('u', 0, 'c', float('nan'))])
        with self.assertRaises(ValueError):
            self.ev.evaluate(self._units(lambda u: [{'baseline': 0.0, 'read_row': 0.0}]),
                             self.CANDS, self.TESTED, 'absolute', tolerance=0.1)

    def test_signed_mode_keeps_an_unbiased_candidate_at_zero_tolerance(self):
        # errors -0.3, -0.1, 0.1, 0.3 in turn: mean exactly zero
        data = self._units(lambda u: [{'probe': 0.2 * (u % 4) - 0.3}], n=40)
        out = self.ev.evaluate(data, {'unbiased': {'probe': 0.0}, 'biased': {'probe': 1.0}},
                               ['probe'], 'signed', tolerance=0.0)
        self.assertNotEqual(out['status']['unbiased'], 'excluded')
        self.assertEqual(out['status']['biased'], 'excluded')

    def test_compare_needs_no_tolerance_and_reports_both_tests(self):
        out = self.ev.compare(self.data, self.candidates, self.TESTED, scope='pooled')
        c = out[0]
        self.assertEqual(c['units_favouring'], {'A_visible_read': 0, 'B_null_read': 32})
        self.assertEqual(c['sign_test_p'], 2 / 2 ** 32)
        self.assertIn('symmetric', c['assumptions']['sign_flip_p'])
        with self.assertRaises(ValueError):
            self.ev.compare(self.data, {'A': self.candidates['A_visible_read']}, self.TESTED)

    def test_exact_sign_flip_for_small_samples(self):
        from causal_decidability.evaluate import _sign_flip_p
        import random as _r
        self.assertAlmostEqual(_sign_flip_p([1.0] * 10, 0, _r.Random(0)), 2 / 1024)


class TwelveCells(unittest.TestCase):

    def setUp(self):
        self.pred = twelve.table()

    def test_108_entries_nine_candidates_seven_signatures(self):
        self.assertEqual(sum(len(row) for row in self.pred.values()), 108)
        self.assertEqual(len(cd.signatures(self.pred)), 7)
        self.assertIn(sorted(twelve.SHIFTS), cd.signatures(self.pred))

    def test_hand_checks(self):
        index = twelve.CELLS.index(('D3', 'R1'))
        self.assertEqual(self.pred['V transfer Object value'][index], 0.0)
        self.assertEqual(self.pred['S30 shift point difference by 30'][index], 1.0)
        index = twelve.CELLS.index(('D4', 'R1'))
        self.assertEqual(self.pred['W transfer Alternative value'][index], 0.0)
        self.assertEqual(self.pred['D copy donor choice'][index], 1.0)

    def test_six_cells_collapse_and_each_donor_buys_one_distinction(self):
        six = twelve.cells_of('D1', 'D2')
        self.assertEqual(len(cd.signatures(cd.restrict(self.pred, six))), 5)
        d3 = {p for p in cd.gains(self.pred, six, twelve.cells_of('D3'))}
        self.assertEqual({tuple(sorted(p)) for p in d3},
                         {tuple(sorted((s, 'V transfer Object value'))) for s in twelve.SHIFTS})
        d4 = cd.gains(self.pred, six, twelve.cells_of('D4'))
        self.assertEqual(d4, [('D copy donor choice', 'W transfer Alternative value')])

    def test_compatible_set_keeps_the_shift_group_whole(self):
        out = cd.compatible_set(self.pred, self.pred[twelve.SHIFTS[1]], 0.0,
                                equivalence_groups=[twelve.SHIFTS])
        self.assertEqual(out['retained'], sorted(twelve.SHIFTS))
        self.assertEqual(out['outcome'], 'resolved')


class Command(unittest.TestCase):

    def _run(self, *args):
        return subprocess.run([sys.executable, '-m', 'causal_decidability', *args],
                              cwd=ROOT, capture_output=True, text=True,
                              env={'PYTHONPATH': str(ROOT / 'src'), 'PATH': ''})

    def test_exit_status_follows_the_verdict(self):
        ok = self._run('--prediction', 'a=0', '--prediction', 'b=1', '--n', '100',
                       '--sigma', '0.1')
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertIn('DECIDABLE', ok.stdout)
        no = self._run('--prediction', 'a=0', '--prediction', 'b=0.01', '--n', '4',
                       '--sigma', '1')
        self.assertEqual(no.returncode, 1)
        bad = self._run('--prediction', 'a=0', '--prediction', 'b=1', '--n', '10',
                        '--sigma', 'nan')
        self.assertEqual(bad.returncode, 2)
        self.assertIn('finite', bad.stderr)

    def test_json_output(self):
        out = self._run('--prediction', 'a=0', '--prediction', 'b=1', '--n', '100',
                        '--sigma', '0.1', '--json')
        result = json.loads(out.stdout)
        self.assertTrue(math.isfinite(result['ratio']))


if __name__ == '__main__':
    unittest.main()
