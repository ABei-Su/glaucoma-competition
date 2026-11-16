import unittest
import numpy as np
import torch
from PIL import Image
from robustness import StressTransform,choose_robust_candidate,conservative_upgrade


class RobustnessTests(unittest.TestCase):
    def test_small_or_uncertain_clean_gain_is_not_promoted(self):
        self.assertFalse(conservative_upgrade(.002,-.0001))
        self.assertFalse(conservative_upgrade(.002,0.))
        self.assertFalse(conservative_upgrade(.0004,.0001))
        self.assertTrue(conservative_upgrade(.002,.0001))
        with self.assertRaises(ValueError):conservative_upgrade(float('nan'),.0001)

    def conditions(self,clean,dim,saturation):
        return {name:dict(auroc=value,average_precision=.8) for name,value in zip(('clean','dim','desaturated'),(clean,dim,saturation))}

    def test_clean_gain_alone_and_stress_gain_alone_do_not_qualify(self):
        reference=self.conditions(.99,.98,.985)
        candidates=dict(plain=self.conditions(.993,.979,.988),tta=self.conditions(.9901,.985,.988))
        self.assertIsNone(choose_robust_candidate(reference,candidates,dict(plain=True,tta=True),8)['selected'])

    def test_resource_filter_worst_condition_ranking_and_frozen_epoch(self):
        reference=self.conditions(.99,.98,.985)
        candidates=dict(plain=self.conditions(.992,.983,.987),tta=self.conditions(.993,.984,.988))
        report=choose_robust_candidate(reference,candidates,dict(plain=True,tta=False),6)
        self.assertEqual(report['selected']['mode'],'plain');self.assertEqual(report['selected']['epoch'],6)
        report=choose_robust_candidate(reference,candidates,dict(plain=True,tta=True),6)
        self.assertEqual(report['selected']['mode'],'tta')
        with self.assertRaises(ValueError):choose_robust_candidate(reference,candidates,dict(plain=True,tta=True),0)

    def test_stress_transform_is_deterministic_without_mutating_input(self):
        image=Image.fromarray(np.tile(np.array([160,90,40],dtype=np.uint8),(32,32,1)))
        before=np.array(image).copy()
        for mode in ('clean','dim','desaturated'):
            transform=StressTransform(mode);a=transform(image);b=transform(image)
            self.assertTrue(torch.equal(a,b));self.assertTrue(torch.isfinite(a).all())
        np.testing.assert_array_equal(np.array(image),before)
        with self.assertRaises(ValueError):StressTransform('arbitrary')
