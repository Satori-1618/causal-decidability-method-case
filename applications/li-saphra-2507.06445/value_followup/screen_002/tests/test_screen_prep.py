"""Input/screen bookkeeping tests; no model checkpoint execution."""
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from prepare_screen import draw_candidates,extended_exclusions
from run_screen import select_indices


class ScreenPreparationTests(unittest.TestCase):
    def test_signed_threshold_and_first_in_order(self):
        result=select_indices([9.,-20.,8.,7.,2.,10.,6.],per_stratum=2)
        self.assertEqual(result,{'accepted':[1,3],'rejected':[0,2]})

    def test_candidate_generation_repeatable_and_input_only(self):
        exclusions={'recipient_strings':[]}
        a=draw_candidates(exclusions,seed=105,n=8)
        b=draw_candidates(exclusions,seed=105,n=8)
        self.assertEqual(a,b)
        self.assertEqual([row['candidate_index'] for row in a],list(range(8)))
        self.assertEqual(len({row['candidate_id'] for row in a}),8)

    def test_both_prefixes_from_all_prior_donor_and_recipient_inputs(self):
        import json
        value=HERE.parent
        rows=[json.loads(line) for line in (value/'inputs/development_001/cases.jsonl').read_text().splitlines()]
        exclusions=extended_exclusions()
        for family in rows:
            for text in [family['recipient']]+[d['string'] for d in family['donors']]:
                self.assertIn(text,exclusions['recipient_strings'])
                for position in [20,28]:
                    self.assertIn(text[:position],exclusions['prefixes'][str(position)])


if __name__=='__main__':
    unittest.main()
