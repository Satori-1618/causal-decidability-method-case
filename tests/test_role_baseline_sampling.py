"""Token-only Stage A sampling preserves all native rows and IID family units."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

SRC = Path(__file__).resolve().parents[1]/'applications/makelov-2311.17030/src'
sys.path.insert(0, str(SRC))
spec = importlib.util.spec_from_file_location('role_baseline_sampling', SRC/'role_baseline_sampling.py')
sampling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampling)
import role_design


class ToyTokenizer:
    bos_token_id = 0
    name_ids = {'Ada': 10, 'Bea': 11, 'Cal': 12, 'Dee': 13}

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [self.name_ids.get(word, 100+sum(map(ord, word)))
                for word in re.findall(r'\w+|[^\w\s]', text)]


class QueryWidthTokenizer(ToyTokenizer):
    def encode(self, text, add_special_tokens=False):
        ids = super().encode(text, add_special_tokens)
        if ' camera ' in text and 'person who watched' in text:
            ids.append(999)
        return ids


class CollisionWidthTokenizer(ToyTokenizer):
    def encode(self, text, add_special_tokens=False):
        ids = super().encode(text, add_special_tokens)
        if 'received a camera from' in text and text.endswith(' was'):
            ids.append(999)
        return ids


class WrongNameTokenizer(ToyTokenizer):
    def encode(self, text, add_special_tokens=False):
        ids = super().encode(text, add_special_tokens)
        return [99 if token == 10 else token for token in ids] if text.startswith('Then,') else ids


class RoleBaselineSampling(unittest.TestCase):
    def source(self, root):
        (root/'data').mkdir()
        values = {'names': ['old0', 'old1', 'old2', 'old3', 'Ada', 'Bea', 'Cal', 'Dee'],
                  'objects': ['old0', 'old1', 'old2', 'camera', 'cup', 'socks']}
        for key, value in values.items():
            (root/f'data/{key}.json').write_text(json.dumps(value))

    def sample(self, root, tokenizer=None, **kwargs):
        return sampling.sample_families(root, tokenizer or ToyTokenizer(),
                                       **{'n': 8, 'tokenizer_fingerprint': 'toy-v1', **kwargs})

    def test_all_48_native_rows_and_correct_roles_are_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.source(root)
            families, manifest = self.sample(root)
        self.assertEqual(manifest['rows_per_family'], 48)
        self.assertEqual(manifest['n_prompt_rows'], 8*48)
        for family in families:
            self.assertEqual(len(family['rows']), 48)
            self.assertEqual([row['row_id'] for row in family['rows']], sampling.expected_row_ids())
            groups = {}
            rows = {row['row_id']: row for row in family['rows']}
            for row in family['rows']:
                context = sampling.CONTEXTS[row['context_id']]
                answer = context['binding'][role_design.ROLES.index(row['query'])]
                self.assertEqual(row['correct_name_index'], answer)
                self.assertEqual(row['correct_answer_token_id'], family['answer_token_ids'][answer])
                self.assertTrue(row['prompt'].startswith('Then, '))
                self.assertEqual(row['position'], len(row['token_ids'])-1)
                self.assertEqual(row['token_ids'][0], 0)
                self.assertEqual(len(row['name_token_positions']), 3)
                self.assertEqual([row['token_ids'][p] for p in row['name_token_positions']],
                                 [family['answer_token_ids'][i] for i in row['first_mention_name_indices']])
                groups.setdefault((row['wording'], row['context_id']), []).append(row)
            self.assertEqual(len(groups), 16)
            self.assertTrue(all(len({len(row['token_ids']) for row in group}) == 1 for group in groups.values()))
            for wording in role_design.WORDINGS:
                for query in role_design.ROLES:
                    # Same text under distinct scientific context IDs is retained.
                    self.assertEqual(rows[f'{wording}/d0/{query}']['prompt'],
                                     rows[f'{wording}/b0_gave/{query}']['prompt'])
                first, second = rows[f'{wording}/d0/giver'], rows[f'{wording}/d1/receiver']
                self.assertEqual(first['name_token_positions'], second['name_token_positions'])
                self.assertEqual(len(first['token_ids']), len(second['token_ids']))
                self.assertEqual(first['correct_answer_token_id'], second['correct_answer_token_id'])
            self.assertEqual(sampling.validate_family(family), family)

    def test_different_forms_record_their_own_final_positions_without_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.source(root)
            families, _ = self.sample(root, n=1)
        rows = {row['row_id']: row for row in families[0]['rows']}
        self.assertNotEqual(rows['heldout/b0_gave/giver']['position'], rows['heldout/b0_watched/giver']['position'])
        self.assertTrue(all(row['token_ids'].count(0) == 1 for row in rows.values()))

    def test_source_partition_whitelist_slot_weights_and_hashes_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.source(root)
            (root/'data/objects.json').write_text(json.dumps(
                ['old0', 'old1', 'old2', 'old3', 'camera', 'camera', 'cup', 'socks']))
            families, manifest = self.sample(root)
            definition = manifest['sampling_definition']
            self.assertEqual(definition['axes']['names'], ['Ada', 'Bea', 'Cal', 'Dee'])
            self.assertEqual(definition['axes']['objects'], ['camera', 'camera', 'cup'])
            self.assertEqual(definition['proposal_tuple_count'], 4*3*2*3)
            self.assertEqual(definition['object_whitelist'], list(sampling.OBJECT_WHITELIST))
            self.assertNotIn('socks', [c['item'] for c in families])
            for name, digest in definition['source_files'].items():
                self.assertEqual(digest, hashlib.sha256((root/name).read_bytes()).hexdigest())
            self.assertEqual(set(definition['code_files_sha256']),
                             {'role_baseline_sampling.py', 'role_design.py', 'donor_factor_sampling.py'})
            self.assertEqual(manifest['mode'], 'native_baseline_only')
            self.assertIn('no model output', manifest['outcome_filter'])
            self.assertIsNone(definition['eligible_population_count'])

    def test_seed_repeats_are_deterministic_but_remain_independent_family_draws(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.source(root)
            # Six ordered triples and one object force repeated content.
            (root/'data/names.json').write_text(json.dumps(['old0', 'old1', 'old2', 'Ada', 'Bea', 'Cal']))
            (root/'data/objects.json').write_text(json.dumps(['old', 'cup']))
            families, manifest = self.sample(root, n=32)
            self.assertEqual((families, manifest), self.sample(root, n=32))
            self.assertEqual(len({f['case_id'] for f in families}), 32)
            self.assertTrue(all(f['case_id'] == f['unique_draw_id'] for f in families))
            self.assertEqual([f['draw_index'] for f in families], list(range(32)))
            self.assertLessEqual(manifest['unique_content_families'], 6)
            self.assertGreaterEqual(manifest['repeated_content_draws'], 26)
            changed, other = self.sample(root, n=32, stage='different-declared-stage')
            self.assertEqual([f['content_family_id'] for f in families], [f['content_family_id'] for f in changed])
            self.assertNotEqual(families[0]['case_id'], changed[0]['case_id'])
            duplicate = next(f for f in families[1:] if f['content_family_id'] == families[0]['content_family_id'])
            families[0]['rows'][0]['token_ids'][0] = 999
            self.assertEqual(duplicate['rows'][0]['token_ids'][0], 0)

    def test_exclusion_of_a_late_heldout_observer_row_rejects_the_whole_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.source(root)
            first, _ = self.sample(root, n=1)
            excluded = first[0]['rows'][-1]['prompt']
            families, manifest = self.sample(root, excluded_prompts=[excluded])
            self.assertGreaterEqual(manifest['rejection_counts']['excluded_prompt'], 1)
            self.assertTrue(all(excluded not in f['prompts'] for f in families))
            self.assertTrue(all(f['content_family_id'] != first[0]['content_family_id'] for f in families))
            (root/'data/objects.json').write_text(json.dumps(['old', 'socks']))
            with self.assertRaisesRegex(ValueError, 'no object slots'):
                self.sample(root)

    def test_all_query_lengths_name_tokens_and_donor_collision_must_qualify(self):
        for tokenizer, reason in ((QueryWidthTokenizer(), 'unequal_query_lengths'),
                                  (CollisionWidthTokenizer(), 'donor_collision_alignment'),
                                  (WrongNameTokenizer(), 'name_token_alignment')):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.source(root)
                families, manifest = self.sample(root, tokenizer)
                self.assertGreater(manifest['rejection_counts'][reason], 0)
                if reason == 'name_token_alignment':
                    self.assertTrue(all('Ada' not in f['names'] for f in families))
                else:
                    self.assertTrue(all(f['item'] == 'cup' for f in families))
                self.assertEqual(manifest['proposal_count'], len(families)+sum(manifest['rejection_counts'].values()))

    def test_historical_exclusions_require_both_3a_stages_and_all_earlier_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, name in enumerate(sampling.HISTORICAL_CASE_FILES):
                path = root/name
                path.parent.mkdir(parents=True)
                size = 4 if name in sampling.ROUND3A_CASE_FILES else 2
                path.write_text(json.dumps([{'prompts': [f'old {index} row {j}' for j in range(size)]}]))
            excluded, provenance = sampling.historical_exclusions(root)
            self.assertEqual(len(excluded), 16)
            self.assertEqual({p['path'] for p in provenance['sources']}, set(sampling.HISTORICAL_CASE_FILES))
            self.assertEqual(provenance['optional_sources_absent'], list(sampling.OPTIONAL_HISTORICAL_FILES))
            for entry in provenance['sources']:
                self.assertEqual(entry['sha256'], hashlib.sha256((root/entry['path']).read_bytes()).hexdigest())
            (root/sampling.ROUND3A_CASE_FILES[-1]).unlink()
            with self.assertRaises(FileNotFoundError):
                sampling.historical_exclusions(root)

    def test_missing_rows_mislabeled_answers_and_modified_prompt_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.source(root)
            original = self.sample(root, n=1)[0][0]
        for mutation in ('omission', 'answer', 'position', 'text', 'name_slot'):
            family = copy.deepcopy(original)
            if mutation == 'omission':
                family['rows'].pop()
                family['prompts'].pop()
            elif mutation == 'answer':
                family['rows'][0]['correct_name_index'] = (family['rows'][0]['correct_name_index']+1)%3
            elif mutation == 'position':
                family['rows'][0]['position'] -= 1
            elif mutation == 'text':
                family['rows'][0]['prompt'] += ' added name Ada'
                family['prompts'][0] = family['rows'][0]['prompt']
            else:
                row = family['rows'][0]
                row['token_ids'][row['name_token_positions'][0]] = 999
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                sampling.validate_family(family)


if __name__ == '__main__':
    unittest.main()
