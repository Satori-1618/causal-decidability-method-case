"""Position, unpadded batching and fail-closed replay of the native-only runner."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / 'applications/makelov-2311.17030'
sys.path.insert(0, str(SOURCE / 'src'))
spec = importlib.util.spec_from_file_location('role_baseline_runner',
                                            SOURCE / 'scripts/run_role_baseline.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'native runner tests require optional torch runtime')
class NativeRunnerTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for index, ids in enumerate(([0, 4, 5], [0, 4, 6], [0, 4, 5, 6])):
            rows.append({'row_id': str(index), 'prompt': str(index), 'token_ids': ids,
                         'position': len(ids)-1, 'wording': 'fit', 'context_id': 'd0',
                         'form': 'gave', 'query': 'giver', 'correct_name_index': 0})
        case = {'rows': rows, 'bos_token_id': 0, 'answer_token_ids': [1, 2, 3]}

        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return rows[int(text)]['token_ids'][1:]

            def decode(self, ids):
                return f'token:{ids[0]}'

        class Model:
            def __init__(self):
                self.calls = []

            def __call__(self, tokens):
                self.calls.append(tokens.tolist())
                vocab = torch.arange(8, dtype=torch.float32)
                # Logit gaps depend on each absolute input position and its token.
                return tokens.float().unsqueeze(-1)*vocab/8

            def run_with_hooks(self, *args, **kwargs):
                raise AssertionError('baseline runner must not use patch hooks')

        return case, Tokenizer(), Model()

    def test_native_final_position_and_no_padding(self):
        case, tokenizer, model = self.fixture()
        with patch('role_baseline_sampling.validate_family'):
            result = runner.measure_family(model, tokenizer, case)
        self.assertTrue(result['controls']['passed'])
        self.assertEqual(result['model_forward_calls'], 3)  # two lengths plus replay
        self.assertEqual(result['prompt_evaluations'], 4)
        self.assertEqual([len(batch) for batch in model.calls], [2, 1, 1])
        self.assertEqual([len(batch[0]) for batch in model.calls], [3, 4, 3])
        for row, observed in zip(case['rows'], result['rows']):
            last = row['token_ids'][row['position']]
            self.assertEqual(observed['name_logits'], [last*i/8 for i in [1, 2, 3]])
            self.assertEqual(observed['full_vocab_argmax_id'], 7)
            self.assertTrue(0 < observed['name_probability_mass'] < 1)

    def test_tokenizer_drift_fails_before_any_model_call(self):
        case, tokenizer, model = self.fixture()
        case['rows'][0]['token_ids'] = [0, 6]
        with patch('role_baseline_sampling.validate_family'), patch.object(
                tokenizer, 'encode', return_value=[4, 5]):
            with self.assertRaisesRegex(RuntimeError, 'tokenizer replay'):
                runner.measure_family(model, tokenizer, case)
        self.assertEqual(model.calls, [])

    def test_bad_replay_fails_closed(self):
        case, tokenizer, native = self.fixture()

        def model(tokens):
            value = native(tokens)
            return value + (.25 if len(native.calls) == 3 else 0.)

        with patch('role_baseline_sampling.validate_family'):
            with self.assertRaisesRegex(RuntimeError, 'replay discrepancy'):
                runner.measure_family(model, tokenizer, case)

    def test_nonfinite_full_logit_blocks_even_if_candidate_names_are_finite(self):
        case, tokenizer, native = self.fixture()

        def model(tokens):
            value = native(tokens)
            value[:, -1, 7] = float('nan')
            return value

        with patch('role_baseline_sampling.validate_family'):
            with self.assertRaisesRegex(RuntimeError, 'nonfinite native logits'):
                runner.measure_family(model, tokenizer, case)


if __name__ == '__main__':
    unittest.main()
