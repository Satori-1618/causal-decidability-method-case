"""Runner bookkeeping tests, without released-model execution."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from design import ANCHORS,predictions
from run import choose_recipient_positions,forecast_precision,structural_receipt


class RunnerTests(unittest.TestCase):
    def test_selection_uses_only_closing_tokens_and_first_tie(self):
        native=SimpleNamespace(attention=torch.tensor([[.1,.9,.3,.8,.3,.1]],dtype=torch.float64))
        self.assertEqual(choose_recipient_positions(['()()'],native),[2])
        native.attention=native.attention.float()
        with self.assertRaises(ValueError):
            choose_recipient_positions(['()()'],native)

    def test_precision_bound_applies_to_target_minus_correct_anchor(self):
        names=list(ANCHORS)+list(predictions({ANCHORS[0]:0.,ANCHORS[1]:1.})['H_state'])
        cells={name:{tag:{'patched_margin':0.} for tag in ['float32','float64']} for name in names}
        cells['neg_28_0']['float32']['patched_margin']=.0007
        cells[ANCHORS[0]]['float32']['patched_margin']=-.0007
        errors=forecast_precision(cells)
        self.assertAlmostEqual(errors['H_state:neg_28_0'],.0014)
        self.assertAlmostEqual(errors['H_position:neg_28_0'],.0007)

    def test_existing_preflight_distinguishes_only_after_crossed_cells(self):
        anchors={name:{'float64':{'patched_margin':value}} for name,value in zip(ANCHORS,[1.,2.])}
        rows=[{'family_id':'x','anchors':anchors,'predictions':predictions({ANCHORS[0]:1.,ANCHORS[1]:2.})}]
        receipt=structural_receipt(rows)
        self.assertEqual(receipt['anchors_only']['pairs'][0]['status'],'identical_declared_means')
        self.assertEqual(receipt['with_declared_target_cells']['pairs'][0]['status'],'different_declared_means')


if __name__=='__main__':
    unittest.main()
