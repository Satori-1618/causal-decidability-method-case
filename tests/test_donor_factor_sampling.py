"""Round-3A sampling fixes the whole family before inspecting any model result."""
import copy
import hashlib
import importlib.util
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'applications/makelov-2311.17030/src/donor_factor_sampling.py'
spec = importlib.util.spec_from_file_location('donor_factor_sampling', MODULE)
sampling = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sampling
spec.loader.exec_module(sampling)


class ToyTokenizer:
    bos_token_id = 0
    name_ids = {'Ada': 10, 'Bea': 11, 'Cal': 12}

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [self.name_ids.get(word, 100 + sum(map(ord, word)))
                for word in re.findall(r'\w+|[^\w\s]', text)]


class UnequalWidthTokenizer(ToyTokenizer):
    def encode(self, text, add_special_tokens=False):
        ids = super().encode(text, add_special_tokens)
        if text.startswith('Then, Bea and Ada'):
            ids.append(999)
        return ids


class WrongNameTokenTokenizer(ToyTokenizer):
    def encode(self, text, add_special_tokens=False):
        ids = super().encode(text, add_special_tokens)
        # Single-name eligibility passes, but the in-context name id is wrong.
        if 'Then,' in text:
            ids = [99 if token == 10 else token for token in ids]
        return ids


class ChangedScaffoldTokenizer(ToyTokenizer):
    def encode(self, text, add_special_tokens=False):
        ids = super().encode(text, add_special_tokens)
        # Names align correctly, but a later scaffold token depends on their order.
        if text.startswith('Then, Bea and Ada') and text.endswith('room'):
            ids[-1] = 999
        return ids


class DonorFactorSampling(unittest.TestCase):
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

    def sample(self, root, tokenizer=None, **kwargs):
        return sampling.sample_iid_families(
            root, tokenizer or ToyTokenizer(),
            **{'n': 32, 'seed': 13, 'stage': 'development',
               'tokenizer_fingerprint': 'toy-v1', **kwargs})

    def test_factorial_cells_align_both_recipient_panels_and_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.source(root)
            draws, manifest = self.sample(root)
        self.assertEqual(manifest['n_family_draws'], 32)
        for case in draws:
            self.assertEqual(case['patterns'], ['ABB', 'BAB', 'BAA', 'ABA'])
            self.assertEqual(case['recipient_indices'], [0, 1])
            self.assertEqual(case['donor_indices'],
                             [{'00': 0, '01': 1, '10': 2, '11': 3},
                              {'00': 1, '01': 0, '10': 3, '11': 2}])
            for r, indices in zip(case['recipient_indices'], case['donor_indices']):
                receiver = case['prompt_roles'][r]
                self.assertEqual(receiver['correct_answer'], case['io'])
                self.assertEqual(indices['00'], r)
                for cell, donor_index in indices.items():
                    donor = case['prompt_roles'][donor_index]
                    self.assertEqual(int(donor['correct_answer'] != receiver['correct_answer']),
                                     int(cell[0]))
                    self.assertEqual(int(donor['correct_answer_mention_position'] !=
                                         receiver['correct_answer_mention_position']), int(cell[1]))
                    self.assertNotEqual(donor['repeated_subject'], donor['correct_answer'])
            slots = case['name_token_positions']
            self.assertEqual(len(slots), 3)
            for pattern, ids in zip(case['patterns'], case['token_ids']):
                self.assertEqual(ids[0], 0)
                self.assertEqual(case['position'], len(ids) - 1)
                self.assertEqual([ids[p] for p in slots],
                                 [case['answer_token_ids'][letter == 'B'] for letter in pattern])
            self.assertEqual(sampling.validate_family(case), case)

    def test_iid_repeats_are_retained_as_unique_independent_draws(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.source(root)
            draws, manifest = self.sample(root, n=40)
        self.assertEqual(len({c['case_id'] for c in draws}), 40)
        self.assertTrue(all(c['case_id'] == c['unique_draw_id'] for c in draws))
        self.assertEqual([c['draw_index'] for c in draws], list(range(40)))
        self.assertLessEqual(manifest['unique_content_families'], 6)
        self.assertEqual(manifest['repeated_content_draws'], 40 - manifest['unique_content_families'])
        duplicate = next(c for c in draws[1:] if c['content_family_id'] == draws[0]['content_family_id'])
        original = duplicate['token_ids'][0][0]
        draws[0]['token_ids'][0][0] = 999
        self.assertEqual(duplicate['token_ids'][0][0], original)

    def test_same_seed_is_exactly_reproducible_and_stage_changes_only_draw_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.source(root)
            first, manifest = self.sample(root)
            self.assertEqual((first, manifest), self.sample(root))
            second, other = self.sample(root, stage='confirmation')
            self.assertEqual([c['content_family_id'] for c in first],
                             [c['content_family_id'] for c in second])
            self.assertNotEqual(first[0]['case_id'], second[0]['case_id'])
            self.assertEqual(manifest['population_definition_sha256'],
                             other['population_definition_sha256'])

    def test_excluding_any_donor_discards_the_whole_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.source(root)
            original, _ = self.sample(root, n=1)
            for donor_index in range(4):
                excluded = original[0]['prompts'][donor_index]
                draws, manifest = self.sample(root, excluded_prompts=[excluded])
                self.assertGreater(manifest['rejection_counts']['excluded_prompt'], 0)
                self.assertTrue(all(excluded not in c['prompts'] for c in draws))
                self.assertTrue(all(c['content_family_id'] != original[0]['content_family_id']
                                    for c in draws))
            complete, _ = self.sample(root, n=200)
            excluded = {p for c in complete for p in c['prompts']}
            with self.assertRaisesRegex(ValueError, 'proposal budget exhausted'):
                self.sample(root, n=1, excluded_prompts=excluded, max_proposals=30)

    def test_all_four_prompts_must_have_equal_width_correct_name_ids_and_constant_scaffold(self):
        failures = [(UnequalWidthTokenizer(), 'unequal_token_lengths'),
                    (WrongNameTokenTokenizer(), 'name_token_alignment'),
                    (ChangedScaffoldTokenizer(), 'non_name_token_variation')]
        for tokenizer, reason in failures:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.source(root)
                draws, manifest = self.sample(root, tokenizer)
                self.assertGreater(manifest['rejection_counts'][reason], 0)
                for case in draws:
                    if reason == 'name_token_alignment':
                        self.assertNotIn('Ada', (case['io'], case['subject']))
                    else:
                        self.assertNotEqual(set((case['io'], case['subject'])), {'Ada', 'Bea'})

    def test_historical_round2_archive_and_complete_new_development_exclusions_have_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, name in enumerate(sampling.HISTORICAL_CASE_FILES):
                path = root / name
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps([{'prompts': [f'old {index} A', f'old {index} B']}]))
            archive = root / sampling.OPTIONAL_HISTORICAL_FILES[0]
            archive.parent.mkdir(parents=True)
            archive.write_text(json.dumps({'base_sentence': 'archive A',
                                           'source_sentence': 'archive B'}) + '\n')
            development = [{'prompts': ['new ABB', 'new BAB', 'new BAA', 'new ABA']}]
            excluded, provenance = sampling.historical_exclusions(root, development_cases=development)
            self.assertEqual(len(excluded), 14)
            self.assertEqual(len(provenance['sources']), 5)
            self.assertEqual(provenance['additional_development_draws'], 1)
            self.assertTrue(set(development[0]['prompts']) <= excluded)
            with self.assertRaisesRegex(ValueError, 'all four'):
                sampling.historical_exclusions(root, development_cases=[{'prompts': ['only ABB', 'only BAB']}])
            self.assertEqual(provenance['excluded_prompts_sha256'], sampling._hash(sorted(excluded)))
            for entry in provenance['sources']:
                self.assertEqual(entry['sha256'], hashlib.sha256((root / entry['path']).read_bytes()).hexdigest())
            new_path = root / 'round3a_development.json'
            new_path.write_text(json.dumps(development))
            from_file, file_provenance = sampling.historical_exclusions(root, extra_case_files=[new_path])
            self.assertEqual(from_file, excluded)
            self.assertEqual(len(file_provenance['sources']), 6)
            archive.unlink()
            _, provenance = sampling.historical_exclusions(root)
            self.assertEqual(provenance['optional_sources_absent'], list(sampling.OPTIONAL_HISTORICAL_FILES))
            (root / sampling.HISTORICAL_CASE_FILES[-1]).unlink()
            with self.assertRaises(FileNotFoundError):
                sampling.historical_exclusions(root)

    def test_source_weights_header_bos_and_definition_hash_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.source(root)
            (root / 'data/objects.json').write_text(json.dumps(['old0', 'old1', 'book', 'book']))
            draws, manifest = self.sample(root)
            definition = manifest['sampling_definition']
            self.assertEqual(definition['axes']['objects'], ['book', 'book'])
            self.assertEqual(definition['axes']['prefix'], 'Then, ')
            self.assertEqual(definition['proposal_tuple_count'], 12)
            self.assertEqual(definition['bos_token_id'], ToyTokenizer.bos_token_id)
            self.assertTrue(all(p.startswith('Then, ') for c in draws for p in c['prompts']))
            for name, expected in definition['source_files'].items():
                self.assertEqual(expected, hashlib.sha256((root / name).read_bytes()).hexdigest())
            _, excluded = self.sample(root, excluded_prompts=['not an eligible prompt'])
            _, tokenizer = self.sample(root, tokenizer_fingerprint='toy-v2')
            self.assertEqual(len({m['population_definition_sha256']
                                  for m in (manifest, excluded, tokenizer)}), 3)
            self.assertIn('no outcome filtering', manifest['inference_order'])
            self.assertIsNone(definition['eligible_population_count'])

    def test_incomplete_or_inconsistent_saved_families_fail_without_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.source(root)
            cases, _ = self.sample(root, n=1)
        case = cases[0]
        for key in ('prompt_roles', 'name_token_positions', 'donor_indices', 'token_ids', 'scaffold'):
            incomplete = copy.deepcopy(case)
            del incomplete[key]
            with self.subTest(missing=key), self.assertRaisesRegex(ValueError, 'missing required'):
                sampling.validate_family(incomplete)
        variants = []
        bad = copy.deepcopy(case)
        bad['token_ids'][3][bad['name_token_positions'][2]] = 123
        variants.append(bad)
        bad = copy.deepcopy(case)
        bad['donor_indices'][1] = bad['donor_indices'][0]
        variants.append(bad)
        bad = copy.deepcopy(case)
        bad['prompt_roles'][2]['correct_answer'] = case['io']
        variants.append(bad)
        bad = copy.deepcopy(case)
        bad['position'] = 0
        variants.append(bad)
        bad = copy.deepcopy(case)
        bad['prompts'][2] = 'arbitrary text despite apparently valid token ids'
        variants.append(bad)
        bad = copy.deepcopy(case)
        bad['donor_indices'][0]['01'] = True
        variants.append(bad)
        for bad in variants:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                sampling.validate_family(bad)

    def test_invalid_slot_template_rejected_before_any_case_is_returned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.source(root)
            (root / 'data/templates.json').write_text(json.dumps(
                ['old0', 'old1', '{name_A} and {name_B} meet {name_C} again {name_C}']))
            with self.assertRaisesRegex(ValueError, 'proposal budget exhausted'):
                self.sample(root, n=1, max_proposals=10)


if __name__ == '__main__':
    unittest.main()
