"""Semantic counterexamples and hook integration for paired Round 2 development."""
import copy
import importlib.util
import json
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'applications/gur-arieh-2510.06182'
sys.path.insert(0,str(APP/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from mixing_word_structure import swap_pair, paired_measures, summarize
from mixing_round1_design import random_matrix, design_indices

CELL={'i_P':3,'i_L':5,'i_R':1,'i_N':6}
RULE={'minimum_mean_abs_delta_nats':1.0,'pair_mass_floor':0.1,
      'answer_mass_floor':0.5,'relative_residual_tolerance':0.25}


def world(d0,d1,base0=0,base1=0):
    arms=[]
    for d,b in [(d0,base0),(d1,base1)]:
        baseline=[-8.0]*7; baseline[3]=b/2; baseline[5]=-b/2
        logits=[-8.0]*7;logits[3]=(b+d)/2;logits[5]=-(b+d)/2
        arms.append({'patch':{'answer_logits':logits,'answer_mass_full_vocab':0.99},
                     'native':{'recipient':{'readout':{'answer_logits':baseline}}}})
    return {'case_id':'synthetic','qualifies':True,'cell':CELL,'arms':arms}


class DesignTests(unittest.TestCase):
    def test_swap_preserves_native_answers_queries_and_indices(self):
        pools=[[f'{c}{i}' for i in range(10)] for c in ('m','g','i')]
        for seed in range(30):
            g=random_matrix(7,pools,random.Random(seed))
            a,b=swap_pair(g,CELL)
            self.assertEqual(design_indices(**a),CELL)
            self.assertEqual(design_indices(**b),CELL)
            for role,index in [('recipient',6),('donor',3)]:
                self.assertEqual(a[role][index][1],b[role][index][1])
            self.assertEqual(a['recipient_query'],b['recipient_query'])
            self.assertEqual(a['donor_query'],b['donor_query'])
            self.assertEqual(swap_pair(b['recipient'],CELL)[1]['recipient'],g)
            self.assertEqual(b['recipient'][3][1],a['recipient'][5][1])
            self.assertEqual(b['recipient'][5][1],a['recipient'][3][1])


class CountermodelTests(unittest.TestCase):
    def test_structure_and_word_have_disjoint_predictions(self):
        self.assertEqual(paired_measures(world(4,4),RULE)['label'],'structure_like')
        self.assertEqual(paired_measures(world(4,-4),RULE)['label'],'word_like')

    def test_reverse_orientation_does_not_change_the_result(self):
        for d0,d1 in [(4,4),(4,-4),(3,1),(0,0)]:
            a=paired_measures(world(d0,d1),RULE)
            b=paired_measures(world(-d0,-d1),RULE)
            self.assertEqual(a['label'],b['label'])
            self.assertEqual(a['loss_structure'],b['loss_structure'])
            self.assertEqual(a['loss_word'],b['loss_word'])

    def test_arm_order_does_not_choose_the_winner(self):
        for d0,d1 in [(4,4),(4,-4),(8,2)]:
            self.assertEqual(paired_measures(world(d0,d1),RULE)['label'],
                             paired_measures(world(d1,d0),RULE)['label'])

    def test_no_patch_effect_cannot_be_identified_as_both(self):
        r=paired_measures(world(0,0,base0=8,base1=-8),RULE)
        self.assertEqual(r['label'],'weak_patch_contrast')
        self.assertFalse(r['resolved'])

    def test_additive_baseline_word_bias_is_removed(self):
        plain=paired_measures(world(4,4),RULE)
        biased=paired_measures(world(4,4,base0=9,base1=-9),RULE)
        self.assertEqual(plain['label'],biased['label'])
        self.assertEqual(plain['loss_structure'],biased['loss_structure'])

    def test_same_sign_with_wrong_magnitude_is_not_enough(self):
        self.assertEqual(paired_measures(world(10,2),RULE)['label'],'neither_narrow_profile')

    def test_mixture_is_not_forced_into_one_rival(self):
        self.assertEqual(paired_measures(world(4,0),RULE)['label'],'neither_narrow_profile')

    def test_outside_support_and_answer_mass_are_not_dropped(self):
        for kind in ('support','mass'):
            r=world(4,4)
            if kind=='mass':r['arms'][1]['patch']['answer_mass_full_vocab']=0.1
            else:r['arms'][1]['patch']['answer_logits'][1]=20
            s=summarize([r],RULE)
            self.assertEqual(s['qualified'],1)
            self.assertEqual(s['counts']['outside_pair_support'],1)

    def test_nonfinite_is_a_technical_error(self):
        r=world(4,4);r['arms'][0]['patch']['answer_logits'][3]=float('nan')
        with self.assertRaises(ValueError):paired_measures(r,RULE)

    def test_duplicates_fail(self):
        r=world(4,4)
        with self.assertRaises(ValueError):summarize([r,r],RULE)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),
                     'tiny-model integration needs the existing model environment')
class HookIntegrationTests(unittest.TestCase):
    def test_two_arm_runner_and_identity_on_tiny_gemma(self):
        from mixing_tiny_model import TINY_SPEC,tiny_tokenizer,tiny_model
        from mixing_runner import Runner,round1_pools
        from mixing_word_structure_runner import run_pair
        tokenizer=tiny_tokenizer();model=tiny_model(len(tokenizer),seed=23)
        pools,_,contexts,answers=round1_pools(tokenizer,TINY_SPEC)
        runner=Runner(model,tokenizer,TINY_SPEC,pools,contexts,answers,2)
        matrix=random_matrix(7,[pools[c] for c in TINY_SPEC['categories']],random.Random(9))
        native=runner.native
        def qualified_native(*args,**kwargs):
            prompt,rec,out=native(*args,**kwargs)
            # Random tiny model has no task competence. Override ONLY eligibility
            # to exercise both interventions; not a scientific qualification test.
            rec.update(correct=True,readout_matches_generation=True)
            rec['readout']['answer_mass_full_vocab']=1.0
            return prompt,rec,out
        with patch.object(runner,'native',side_effect=qualified_native):
            record,arrays=run_pair(runner,matrix,CELL,case_id='tiny',seed=9,control_target=3,audit=True)
        self.assertTrue(record['qualifies'])
        self.assertTrue(all(record['technical'].values()))
        self.assertEqual(len(arrays),4)
        self.assertEqual([a['identity_max_full_logit_error'] for a in record['arms']],[0.0,0.0])
        self.assertTrue(record['qualification_before_conflict_patch'])
        self.assertTrue(all(not block._forward_pre_hooks for block in model.model.layers))

    def test_native_failure_skips_conflict_patches(self):
        from mixing_tiny_model import TINY_SPEC,tiny_tokenizer,tiny_model
        from mixing_runner import Runner,round1_pools
        from mixing_word_structure_runner import run_pair
        tokenizer=tiny_tokenizer();model=tiny_model(len(tokenizer),seed=23)
        pools,_,contexts,answers=round1_pools(tokenizer,TINY_SPEC)
        runner=Runner(model,tokenizer,TINY_SPEC,pools,contexts,answers,2)
        matrix=random_matrix(7,[pools[c] for c in TINY_SPEC['categories']],random.Random(9))
        native=runner.native
        def bad_native(*args,**kwargs):
            prompt,rec,out=native(*args,**kwargs);rec['correct']=False
            return prompt,rec,out
        with patch.object(runner,'native',side_effect=bad_native):
            record,_=run_pair(runner,matrix,CELL,case_id='tiny',seed=9,control_target=3)
        self.assertFalse(record['qualifies'])
        self.assertTrue(all('patch' not in arm for arm in record['arms']))


if __name__=='__main__':unittest.main()
