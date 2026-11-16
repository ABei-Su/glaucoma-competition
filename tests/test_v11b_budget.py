import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from torch.utils.data import Dataset
from train import parse_args
try:
    import v11b_protocol as protocol
    from tools import run_v11b
except ModuleNotFoundError:
    protocol=None;run_v11b=None


class Samples(Dataset):
    def __len__(self):return 17
    def __getitem__(self,index):
        y=float(index%2)
        return torch.full((3,32,32),1. if y else -1.),y,str(index)+'.jpg'


class Tiny(nn.Module):
    def __init__(self):super().__init__();self.calls=0
    def forward(self,images):
        self.calls+=1;return images.mean((1,2,3))


class Budgeted448Tests(unittest.TestCase):
    def ready(self):self.assertIsNotNone(protocol,'Budgeted448 variant is missing')

    def test_new_namespace_preserves_training_recipe_and_locked_full_epoch(self):
        self.ready()
        for internal,epoch in [(True,0),(False,5)]:
            expected=protocol.recipe(internal,epoch)
            self.assertIn('v11b_',expected['run']);self.assertEqual(expected['size'],448)
            self.assertEqual(expected['seed'],20261031 if internal else 20261032)
            self.assertEqual(vars(parse_args(protocol.training_command(internal,epoch)[1:])),expected)

    def test_real_bundle_uses_only_r50_tta_and_cn_plain_and_rejects_wrong_pattern(self):
        self.ready()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);first=root/'runs/v3_r50_384_w1/best.pt';first.parent.mkdir(parents=True)
            first.write_bytes(b'fixed r50');second=root/'candidate.pt';second.write_bytes(b'fixed448')
            config=root/'config.json'
            with patch.object(run_v11b,'ROOT',root):identity=run_v11b.bundle(config,second)
            self.assertTrue(identity);spec=json.loads(config.read_text(encoding='utf-8'))
            self.assertEqual([m['tta'] for m in spec['models']],[True,False])
            spec['models'][1]['tta']=True;config.write_text(json.dumps(spec), encoding='utf-8')
            with self.assertRaises(ValueError):protocol.inference_identity(config)

    def test_plain_evaluation_recovers_known_probabilities_with_one_forward_per_batch(self):
        self.ready();model=Tiny()
        with patch.object(run_v11b,'build_model',return_value=model):
            names,labels,probabilities=run_v11b.evaluate_plain(dict(model={}),Samples(),device='cpu',workers=0)
        self.assertEqual(model.calls,2)
        self.assertEqual(names,[str(i)+'.jpg' for i in range(17)])
        self.assertEqual(labels,[float(i%2) for i in range(17)])
        for label,p in zip(labels,probabilities):
            self.assertAlmostEqual(p,1/(1+math.exp(-1 if label else 1)),places=6)


if __name__=='__main__':unittest.main()
