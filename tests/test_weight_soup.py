"""Catch invalid parameter mixing, provenance mismatch and false promotion."""
import copy
import unittest
import torch
from weight_soup import average_pair,validate_pair,soup_gate


def fixture():
    config=dict(architecture='convnext_tiny',pretrain='public.pth',seed=7,size=384,
                data='train',cache='cache384',split_manifest='train-split.json',epochs=8,
                batch=16,workers=4,lr=.0001,pos_weight=1.,memory_format='contiguous',
                validation_probability_precision='fp32',init_checkpoint=None,max_steps=0,
                checkpoint_epoch=0,augmentation='mild')
    checkpoint=dict(architecture='convnext_tiny',size=384,memory_format='contiguous',
                    data_fingerprint='train-only',split_manifest_sha256='fixed-split',
                    validation_partition='internal_train_holdout',
                    validation_probability_precision='fp32',metrics=dict(epoch=8))
    other=copy.deepcopy(config);other['augmentation']='color_robust'
    return config,other,checkpoint,copy.deepcopy(checkpoint)


class SoupTests(unittest.TestCase):
    def test_average_reloads_known_linear_signal_without_source_mutation(self):
        a={'weight':torch.tensor([[2.,4.]]),'bias':torch.tensor([0.])}
        b={'weight':torch.tensor([[6.,0.]]),'bias':torch.tensor([2.])}
        result=average_pair(a,b)
        model=torch.nn.Linear(2,1);model.load_state_dict(result)
        self.assertEqual(model(torch.tensor([[1.,2.]])).item(),9.)
        result['weight'][0,0]=99
        self.assertEqual(a['weight'][0,0].item(),2.)
        self.assertEqual(b['weight'][0,0].item(),6.)
        largest=torch.finfo(torch.float32).max
        self.assertEqual(average_pair({'x':torch.tensor(largest)},{'x':torch.tensor(largest)})['x'].item(),largest)

    def test_rejects_wrong_keys_shapes_dtypes_nonfinite_and_running_stats(self):
        cases=[({'a':torch.zeros(1)},{'b':torch.zeros(1)}),
               ({'a':torch.zeros(1)},{'a':torch.zeros(2)}),
               ({'a':torch.zeros(1)},{'a':torch.zeros(1,dtype=torch.float64)}),
               ({'a':torch.tensor([float('nan')])},{'a':torch.zeros(1)}),
               ({'a':torch.tensor([float('inf')])},{'a':torch.zeros(1)}),
               ({'a':torch.zeros(1,dtype=torch.long)},{'a':torch.zeros(1,dtype=torch.long)}),
               ({'bn.running_mean':torch.zeros(1)},{'bn.running_mean':torch.zeros(1)}),
               ({},{})]
        for a,b in cases:
            with self.subTest(keys=list(a)):
                with self.assertRaises(ValueError):average_pair(a,b)

    def test_pair_rejects_exposed_or_different_initialization_and_accepts_full_pair(self):
        a,b,x,y=fixture();validate_pair(a,b,x,y)
        for key,value in [('seed',8),('pretrain','different.pth'),('init_checkpoint','supervised.pt'),
                          ('split_manifest',None),('augmentation','mild'),('architecture','resnet50')]:
            altered=copy.deepcopy(b);altered[key]=value
            with self.subTest(field=key):
                with self.assertRaises(ValueError):validate_pair(a,altered,x,y)
        for key,value in [('data_fingerprint','exposed'),('split_manifest_sha256',None),
                          ('validation_partition','official_valid'),('metrics',dict(epoch=7))]:
            altered=copy.deepcopy(y);altered[key]=value
            with self.subTest(field=key):
                with self.assertRaises(ValueError):validate_pair(a,b,x,altered)
        a['split_manifest']=b['split_manifest']=None
        a['checkpoint_epoch']=b['checkpoint_epoch']=8
        for saved in [x,y]:
            saved['split_manifest_sha256']=None;saved['validation_partition']='official_valid'
        validate_pair(a,b,x,y)

    def test_gate_requires_both_auroc_improvements_and_non_decreasing_clean_ap(self):
        ref={key:dict(auroc=auc,average_precision=.8) for key,auc in [('clean',.99),('dim',.98),('desaturated',.985)]}
        better=copy.deepcopy(ref)
        for item in better.values():item['auroc']+=.001
        self.assertTrue(soup_gate(ref,better)['passed'])
        for condition,field,value in [('clean','auroc',.9901),('dim','auroc',.9801),('clean','average_precision',.799)]:
            altered=copy.deepcopy(better);altered[condition][field]=value
            self.assertFalse(soup_gate(ref,altered)['passed'])
        with self.assertRaises(ValueError):soup_gate(ref,{'clean':better['clean']})
        altered=copy.deepcopy(better);altered['dim']['auroc']=float('nan')
        with self.assertRaises(ValueError):soup_gate(ref,altered)


if __name__=='__main__':unittest.main()
