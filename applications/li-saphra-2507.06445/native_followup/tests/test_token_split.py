import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import factorized_eos_weights
from token_split import token_factorized_weights


class TokenFactorTests(unittest.TestCase):
    def test_invariants_and_endpoint(self):
        text=['())(', '((((']
        a=torch.tensor([[.10,.30,.10,.05,.25,.20],[.05,.10,.25,.15,.20,.25]],dtype=torch.float64)
        for mode in ['within_token','token_mass','both']:
            b=token_factorized_weights(a,text,mode)
            torch.testing.assert_close(b[:,[0,5]],a[:,[0,5]],rtol=0,atol=0)
            torch.testing.assert_close(b.sum(-1),a.sum(-1))
        w=token_factorized_weights(a,text,'within_token')
        self.assertEqual(float(w[0,1]),float(w[0,4]))
        torch.testing.assert_close(w[0,[1,4]].sum(),a[0,[1,4]].sum())
        m=token_factorized_weights(a,text,'token_mass')
        torch.testing.assert_close(m[0,1]/m[0,4],a[0,1]/a[0,4])
        torch.testing.assert_close(m[0,[1,4]].sum(),a[0,1:5].sum()/2)
        torch.testing.assert_close(token_factorized_weights(a,text,'both'),factorized_eos_weights(a,torch.tensor([5,5]),'routing_only'))

    def test_zero_group_rejected(self):
        a=torch.tensor([[.1,0.,.5,.4]],dtype=torch.float64)
        with self.assertRaises(ValueError):
            token_factorized_weights(a,['()'],'token_mass')


if __name__ == '__main__':
    unittest.main()
