import unittest
from v7_protocol import resource_fallback


class ResourceFallbackTests(unittest.TestCase):
    def record(self):
        first=dict(name='pair_tta',auroc=.992,average_precision=.84,efficientnet_epoch=6)
        second=dict(name='pair_plain',pair=True,tta=False,auroc=.991,average_precision=.83)
        return dict(selected=first,candidates=[first,second],reference=dict(auroc=.99,average_precision=.82))

    def test_only_fixed_second_ranked_pair_plain_and_original_epoch(self):
        source=self.record();chosen=resource_fallback(source)
        self.assertEqual(chosen['name'],'pair_plain');self.assertEqual(chosen['efficientnet_epoch'],6)
        self.assertNotIn('efficientnet_epoch',source['candidates'][1])

    def test_different_ranking_or_non_improving_fallback_is_rejected(self):
        source=self.record();source['candidates'][1]['name']='single_tta'
        with self.assertRaises(ValueError):resource_fallback(source)
        source=self.record();source['candidates'][1]['auroc']=.98
        with self.assertRaises(ValueError):resource_fallback(source)


if __name__=='__main__':unittest.main()
