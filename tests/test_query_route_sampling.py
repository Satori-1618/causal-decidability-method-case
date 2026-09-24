"""Sampling is genuinely with replacement, frozen by inputs, and prompt-disjoint."""
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'applications/makelov-2311.17030/src/query_route_sampling.py'
spec = importlib.util.spec_from_file_location('query_route_sampling', MODULE)
sampling = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sampling
spec.loader.exec_module(sampling)


def eligible_case(tag):
    return {'case_id': 'historical-content-id', 'draw': 42,
            'prompts': [f'{tag} ABB', f'{tag} BAB'], 'patterns': ['ABB', 'BAB'],
            'token_ids': [[0, 10, 11], [0, 11, 10]], 'position': 2,
            'io': 'Ada', 'subject': 'Bea', 'answer_token_ids': [10, 11]}


class ToyTokenizer:
    bos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        if text.strip() in ('Ada', 'Bea', 'Cal'):
            return [{'Ada': 10, 'Bea': 11, 'Cal': 12}[text.strip()]]
        result = [100 + sum(map(ord, word)) for word in text.split()]
        # A format-only failure removes two ordered proposal tuples, never an outcome.
        if text.startswith('Then, Bea and Ada'):
            result.append(999)
        return result


class QueryRouteSampling(unittest.TestCase):
    def source(self, root):
        directory = root / 'data'
        directory.mkdir()
        values = {
            'names': ['oldA', 'oldB', 'oldC', 'Ada', 'Bea', 'Cal'],
            'objects': ['old_object', 'book'], 'places': ['old_place', 'room'],
            'prefixes': ['unused0', 'unused1', 'Then, '],
            'templates': ['unused0', 'unused1',
                          '{name_A} and {name_B} met. {name_C} gives a {object} in {place}'],
        }
        for name, data in values.items():
            (directory / f'{name}.json').write_text(json.dumps(data))

    def test_singleton_pool_retains_duplicates_as_distinct_iid_units(self):
        case = eligible_case('one')
        draws, manifest = sampling.sample_from_eligible_pool(
            [case], n=32, seed=sampling.PILOT_SEED, stage='pilot')
        self.assertEqual(len(draws), 32)
        self.assertEqual(len({c['case_id'] for c in draws}), 32)
        self.assertTrue(all(c['case_id'] == c['unique_draw_id'] for c in draws))
        self.assertEqual(len({c['content_pair_id'] for c in draws}), 1)
        self.assertEqual(manifest['repeated_content_draws'], 31)
        self.assertEqual(manifest['unique_content_pairs'], 1)
        self.assertEqual([c['draw_index'] for c in draws], list(range(32)))
        draws[0]['token_ids'][0][0] = 999
        self.assertEqual(draws[1]['token_ids'][0][0], 0)
        self.assertEqual(case['token_ids'][0][0], 0)

    def test_seed_stage_and_population_are_bound_and_reproducible(self):
        pool = [eligible_case('one'), eligible_case('two')]
        kwargs = dict(n=24, seed=24092402, stage='confirmation')
        draws, manifest = sampling.sample_from_eligible_pool(pool, **kwargs)
        self.assertEqual((draws, manifest), sampling.sample_from_eligible_pool(pool[::-1], **kwargs))
        renamed, other = sampling.sample_from_eligible_pool(pool, **{**kwargs, 'stage': 'other'})
        self.assertEqual([d['content_pair_id'] for d in renamed],
                         [d['content_pair_id'] for d in draws])
        self.assertNotEqual(renamed[0]['case_id'], draws[0]['case_id'])
        self.assertEqual(manifest['population_definition_sha256'], other['population_definition_sha256'])

    def test_either_excluded_prompt_removes_the_entire_pair(self):
        one, two = eligible_case('one'), eligible_case('two')
        draws, _ = sampling.sample_from_eligible_pool(
            [one, two], n=20, seed=3, stage='test', excluded_prompts=[one['prompts'][1]])
        self.assertTrue(all(c['prompts'] == two['prompts'] for c in draws))
        with self.assertRaisesRegex(ValueError, 'empty'):
            sampling.sample_from_eligible_pool([one], n=1, seed=1, stage='test',
                                              excluded_prompts=[one['prompts'][0]])
        with self.assertRaisesRegex(ValueError, 'duplicate content'):
            sampling.sample_from_eligible_pool([one, one], n=1, seed=1, stage='test')

    def test_historical_and_entire_pilot_prompt_sets_are_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, name in enumerate(sampling.HISTORICAL_CASE_FILES):
                path = root / name
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps([eligible_case(f'history{i}')]))
            prior = root / sampling.OPTIONAL_HISTORICAL_FILES[0]
            prior.parent.mkdir(parents=True)
            prior.write_text(json.dumps({'base_sentence': 'table base',
                                         'source_sentence': 'table source'}) + '\n')
            pilot = [eligible_case('pilot1'), eligible_case('pilot1'), eligible_case('pilot2')]
            excluded, provenance = sampling.historical_exclusions(root, pilot_cases=pilot)
            self.assertEqual(len(excluded), 10)
            self.assertEqual(provenance['additional_pilot_draws'], 3)
            self.assertEqual(len(provenance['sources']), 3)
            self.assertEqual(provenance['optional_sources_absent'], [])
            self.assertTrue(set(pilot[2]['prompts']) <= excluded)
            prior.unlink()
            _, incomplete = sampling.historical_exclusions(root)
            self.assertEqual(incomplete['optional_sources_absent'],
                             list(sampling.OPTIONAL_HISTORICAL_FILES))
            (root / sampling.HISTORICAL_CASE_FILES[0]).unlink()
            with self.assertRaises(FileNotFoundError):
                sampling.historical_exclusions(root)

    def test_conditional_rejection_is_format_only_and_keeps_repeated_draws(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.source(root)
            kwargs = dict(n=40, seed=sampling.PILOT_SEED, stage='pilot',
                          tokenizer_fingerprint='toy-v1')
            draws, manifest = sampling.sample_iid_cases(root, ToyTokenizer(), **kwargs)
            self.assertEqual((draws, manifest), sampling.sample_iid_cases(root, ToyTokenizer(), **kwargs))
            self.assertEqual(manifest['sampling_definition']['proposal_tuple_count'], 6)
            self.assertIsNone(manifest['sampling_definition']['eligible_population_count'])
            self.assertGreater(manifest['rejection_counts']['unequal_token_lengths'], 0)
            self.assertGreater(manifest['repeated_content_draws'], 0)
            self.assertEqual(len({c['case_id'] for c in draws}), 40)
            self.assertTrue(all(c['position'] == len(c['token_ids'][0]) - 1 for c in draws))
            self.assertTrue(all(c['io'] != c['subject'] for c in draws))
            # Disjoint confirmation excludes all unique pilot prompts, not just one
            # representative record, and fails if that exhausts this tiny population.
            excluded = {p for d in draws for p in d['prompts']}
            with self.assertRaisesRegex(ValueError, 'proposal budget exhausted'):
                sampling.sample_iid_cases(root, ToyTokenizer(), n=1, seed=8,
                                          stage='confirmation', excluded_prompts=excluded,
                                          tokenizer_fingerprint='toy-v1', max_proposals=40)

    def test_population_definition_changes_with_exclusions_and_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.source(root)
            kwargs = dict(n=2, seed=7, stage='pilot', tokenizer_fingerprint='toy-v1')
            _, original = sampling.sample_iid_cases(root, ToyTokenizer(), **kwargs)
            _, excluded = sampling.sample_iid_cases(root, ToyTokenizer(),
                                                    excluded_prompts=['not a current prompt'], **kwargs)
            _, tokenizer = sampling.sample_iid_cases(root, ToyTokenizer(),
                                                     **{**kwargs, 'tokenizer_fingerprint': 'toy-v2'})
            self.assertEqual(len({m['population_definition_sha256']
                                  for m in (original, excluded, tokenizer)}), 3)
            with self.assertRaisesRegex(ValueError, 'tokenizer_fingerprint'):
                sampling.sample_iid_cases(root, ToyTokenizer(), n=1, seed=7, stage='pilot')

    def test_malformed_case_is_not_silently_repaired(self):
        case = eligible_case('bad')
        for key, value in (('position', 0), ('answer_token_ids', [10, 10]),
                           ('token_ids', [[0, 10], [0, 11, 10]])):
            bad = copy.deepcopy(case)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                sampling.sample_from_eligible_pool([bad], n=1, seed=1, stage='test')


if __name__ == '__main__':
    unittest.main()
