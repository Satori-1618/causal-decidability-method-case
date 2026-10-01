"""No checkpoint execution: forecast and validation bookkeeping only."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from confirm import ARMS,check_token_operator,jsonlines,native_sign_pattern,prediction_error_precision,preflight_receipt
from token_split import token_factorized_weights


class ConfirmationBookkeepingTests(unittest.TestCase):
    def test_preflight_preserves_casewise_opposition_despite_equal_means(self):
        rows = [
            {'predictions': {'a': {'x': 1., 'y': -1.}, 'b': {'x': -1., 'y': 1.}}},
            {'predictions': {'a': {'x': -1., 'y': 1.}, 'b': {'x': 1., 'y': -1.}}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'predictions.jsonl'
            jsonlines(path,rows)
            receipt=preflight_receipt(path,rows,['a','b'],['x','y'])
        self.assertEqual(receipt['preflight']['pairs'][0]['status'],'different_declared_means')
        self.assertEqual(len(receipt['preflight']['identical_mean_groups']),2)

    def test_preflight_keeps_identical_forecasts_together(self):
        rows=[{'predictions':{'a':{'x':2.,'y':2.},'b':{'x':2.,'y':2.}}}]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'predictions.jsonl'
            jsonlines(path,rows)
            receipt=preflight_receipt(path,rows,['a','b'],['x','y'])
        self.assertEqual(receipt['preflight']['pairs'][0]['status'],'identical_declared_means')

    def test_prediction_file_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'predictions.jsonl'
            jsonlines(path,[{'a':1}])
            with self.assertRaises(FileExistsError):
                jsonlines(path,[{'a':2}])
            self.assertEqual(json.loads(path.read_text()),{'a':1})

    def test_sign_pattern_only_depends_on_bracket_ranking(self):
        self.assertTrue(native_sign_pattern(')(', [99.,.1,.4,99.]))
        self.assertTrue(native_sign_pattern(')(', [-99.,.2,.8,-99.]))
        self.assertFalse(native_sign_pattern(')(', [99.,.4,.1,99.]))
        self.assertIsNone(native_sign_pattern('()', [99.,.4,.1,99.]))

    def test_precision_gate_checks_error_contrasts_not_only_arms(self):
        results={tag:{arm:SimpleNamespace(margins=torch.tensor([0.],dtype=torch.float64)) for arm in ARMS}
                 for tag in ['float32','float64']}
        results['float32']['routing_only'].margins[0]=.0007
        results['float32']['both'].margins[0]=-.0007
        self.assertAlmostEqual(prediction_error_precision(results)['routing_only minus both'],.0014)

    def test_delivered_token_contract_accepts_valid_and_rejects_modified_tensor(self):
        strings=['(()())']
        attention=torch.tensor([[.1,.03,.07,.2,.1,.15,.25,.1]],dtype=torch.float64)
        native=SimpleNamespace(attention=attention,eos_positions=torch.tensor([7]))
        for mode in ['within_token','token_mass']:
            intended=token_factorized_weights(attention,strings,mode)
            result=SimpleNamespace(attention=intended.clone())
            report=check_token_operator(native,result,intended,strings,mode)
            self.assertTrue(report['injected_attention_exactly_requested'])
            result.attention[0,0] += .001
            with self.assertRaises(AssertionError):
                check_token_operator(native,result,intended,strings,mode)


if __name__=='__main__':
    unittest.main()
