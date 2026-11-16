import unittest
import numpy as np
from v7_protocol import rank_candidates


class V7ProtocolTests(unittest.TestCase):
    def test_only_measured_eligible_shapes_can_be_selected(self):
        labels=[0,1,0,1]
        new=[.05,.9,.1,.8];reference=[.8,.7,.2,.4]
        allowed={'single_plain':True,'single_tta':False,'pair_plain':False,'pair_tta':False}
        result=rank_candidates(labels,new,new,reference,reference,allowed)
        self.assertEqual(result['selected']['name'],'single_plain')
        self.assertGreater(result['selected']['auroc'],result['reference']['auroc'])
        self.assertEqual(len(result['candidates']),1)

    def test_no_improvement_or_no_feasible_shape_is_not_promoted(self):
        labels=[0,1,0,1];good=[.05,.9,.1,.8];bad=[.8,.7,.2,.4]
        allowed={key:True for key in ['single_plain','single_tta','pair_plain','pair_tta']}
        self.assertIsNone(rank_candidates(labels,bad,bad,good,good,allowed)['selected'])
        self.assertIsNone(rank_candidates(labels,good,good,bad,bad,{key:False for key in allowed})['selected'])

    def test_shape_and_eligibility_validation(self):
        allowed={'single_plain':True,'single_tta':True,'pair_plain':True,'pair_tta':True}
        with self.assertRaises(ValueError):rank_candidates([0,1],[.1], [.1,.9],[.1,.9],[.1,.9],allowed)
        with self.assertRaises(ValueError):rank_candidates([0,1],[.1,.9],[.1,.9],[.1,.9],[.1,.9],{'extra':True})
