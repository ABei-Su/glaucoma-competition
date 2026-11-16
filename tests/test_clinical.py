"""Catch leakage in fitted age transforms, bad joins and incorrect logit fusion."""
import copy
import unittest
import csv
import tempfile
from pathlib import Path
from PIL import Image
import torch
from clinical import parse_age,fit_age_stats,encode_age,AgeConvNeXt,validate_age_stats,AgeFundusDataset,age_gate


class ClinicalTests(unittest.TestCase):
    def test_age_gate_requires_clean_improvement_ap_and_missing_fallback(self):
        ref=dict(auroc=.99,average_precision=.8)
        self.assertTrue(age_gate(ref,dict(auroc=.99001,average_precision=.8),dict(auroc=.9895))['passed'])
        for clean,missing in [(dict(auroc=.99,average_precision=.81),.99),
                              (dict(auroc=.991,average_precision=.79),.99),
                              (dict(auroc=.991,average_precision=.81),.988)]:
            self.assertFalse(age_gate(ref,clean,dict(auroc=missing))['passed'])
        with self.assertRaises(ValueError):age_gate(ref,dict(auroc=float('nan'),average_precision=.8),dict(auroc=.99))

    def test_dataset_keeps_age_label_and_image_join_after_holdout_reordering(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'train').mkdir()
            rows=[dict(Image_name='a.png',Age='20',True_Label='0'),dict(Image_name='b.png',Age='80',True_Label='1')]
            with (root/'train.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['Image_name','Age','True_Label']);writer.writeheader();writer.writerows(rows)
            Image.new('RGB',(32,32),(30,30,30)).save(root/'train/a.png')
            Image.new('RGB',(32,32),(200,200,200)).save(root/'train/b.png')
            stats=fit_age_stats(rows)
            dataset=AgeFundusDataset(root,'train',32,None,list(reversed(rows)),stats)
            _,age,label,name=dataset[0]
            self.assertEqual(name,'b.png');self.assertEqual(label.item(),1.)
            self.assertEqual(age.tolist(),[1.,0.])
            dataset.force_missing=True
            self.assertEqual(dataset[0][1].tolist(),[0.,1.])

    def test_age_parsing_and_training_only_transform_with_missing_values(self):
        for text in ('',None,'nan','inf','-1','121','unknown'):
            self.assertIsNone(parse_age(text))
        self.assertEqual(parse_age(' 42.5 '),42.5)
        self.assertEqual(parse_age('0'),0.)
        stats=fit_age_stats([{'Age':str(age)} for age in (20,40,60,80)])
        self.assertEqual(stats['mean'],50.)
        self.assertAlmostEqual(stats['scale'],500**.5)
        before=copy.deepcopy(stats)
        self.assertEqual(encode_age(None,stats).tolist(),[0.,1.])
        self.assertEqual(encode_age('50',stats).tolist(),[0.,0.])
        self.assertGreater(encode_age('100',stats)[0].item(),2.)
        self.assertEqual(stats,before)
        with self.assertRaises(ValueError):fit_age_stats([{'Age':'30'}]*4)
        with self.assertRaises(ValueError):fit_age_stats([{'Age':'20'},{'Age':'80'},{'Age':''}])

    def test_invalid_saved_transforms_are_rejected_before_encoding(self):
        stats=fit_age_stats([{'Age':'20'},{'Age':'80'}])
        for key,value in [('mean',float('nan')),('scale',0.),('scale',float('inf')),('clip_z',8),('schema_version',99)]:
            changed=dict(stats);changed[key]=value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):validate_age_stats(changed)

    def test_zero_age_branch_preserves_image_logits_and_fixed_age_effect(self):
        base=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3,1))
        model=AgeConvNeXt(base)
        images=torch.randn(2,3,1,1)
        ages=torch.tensor([[-1.,0.],[1.,1.]])
        torch.testing.assert_close(model(images,ages),base(images).flatten(),rtol=0,atol=0)
        with torch.no_grad():model.age.weight.copy_(torch.tensor([[2.,3.]]))
        torch.testing.assert_close(model(images,ages)-base(images).flatten(),torch.tensor([-2.,5.]))
        with self.assertRaises(ValueError):model(images,torch.zeros(2,1))

    def test_branch_learns_known_age_signal_and_reloads_exactly(self):
        base=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(1,1,bias=False))
        with torch.no_grad():base[1].weight.zero_()
        base.requires_grad_(False)
        model=AgeConvNeXt(base)
        optimizer=torch.optim.SGD(model.age.parameters(),lr=.5)
        images=torch.zeros(8,1,1,1)
        ages=torch.tensor([[-1.,0.],[1.,0.]]*4)
        labels=torch.tensor([0.,1.]*4)
        for _ in range(20):
            optimizer.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(model(images,ages),labels)
            loss.backward();optimizer.step()
        p=model(images,ages).sigmoid()
        self.assertLess(p[0].item(),.2);self.assertGreater(p[1].item(),.8)
        restored=AgeConvNeXt(copy.deepcopy(base));restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored(images,ages),model(images,ages),rtol=0,atol=0)


if __name__=='__main__':unittest.main()
