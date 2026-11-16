import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

import model
from predict import Predictor


class InferenceFusionTests(unittest.TestCase):
    def test_nested_eval_fusion_preserves_outputs_and_original_state(self):
        torch.manual_seed(45)
        network=nn.Sequential(nn.Sequential(nn.Conv2d(3,8,3,padding=1,bias=False),nn.BatchNorm2d(8),nn.SiLU()),
                              nn.Conv2d(8,4,1),nn.BatchNorm2d(4)).eval()
        for layer in network.modules():
            if isinstance(layer,nn.BatchNorm2d):
                layer.running_mean.uniform_(-1,1);layer.running_var.uniform_(.3,2)
                layer.weight.data.uniform_(.3,2);layer.bias.data.uniform_(-1,1)
        before=copy.deepcopy(network.state_dict());inputs=torch.randn(2,3,19,17)
        fused,count=model.fuse_inference_batchnorm(network)
        self.assertEqual(count,2)
        self.assertFalse(any(isinstance(x,nn.BatchNorm2d) for x in fused.modules()))
        torch.testing.assert_close(network(inputs),fused(inputs),rtol=2e-4,atol=2e-5)
        for key,value in before.items():self.assertTrue(torch.equal(value,network.state_dict()[key]))

    def test_training_or_stateless_norm_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'eval'):
            model.fuse_inference_batchnorm(nn.Sequential(nn.Conv2d(3,4,1),nn.BatchNorm2d(4)))
        network=nn.Sequential(nn.Conv2d(3,4,1),nn.BatchNorm2d(4,track_running_stats=False)).eval()
        with self.assertRaisesRegex(ValueError,'statistics'):model.fuse_inference_batchnorm(network)

    def test_explicit_efficientnet_bundle_fusion_and_legacy_default(self):
        torch.set_num_threads(2);torch.manual_seed(46)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            network=model.build_model(architecture='efficientnet_v2_s').eval()
            torch.save(dict(model=network.state_dict(),architecture='efficientnet_v2_s',size=64),root/'best.pt')
            configuration=dict(schema_version=1,models=[dict(checkpoint='best.pt',fuse_conv_bn=True)])
            (root/'config.json').write_text(json.dumps(configuration))
            fused=Predictor(root/'config.json','cpu','fp32')
            original=Predictor(root/'best.pt','cpu','fp32')
            self.assertFalse(any(isinstance(x,nn.BatchNorm2d) for x in fused.model.modules()))
            self.assertTrue(any(isinstance(x,nn.BatchNorm2d) for x in original.model.modules()))
            inputs=torch.randn(1,3,64,64)
            with torch.inference_mode():
                torch.testing.assert_close(original.model(inputs),fused.model(inputs),rtol=2e-4,atol=2e-5)
            configuration['models'][0]['fuse_conv_bn']='yes'
            (root/'config.json').write_text(json.dumps(configuration))
            with self.assertRaisesRegex(ValueError,'boolean'):Predictor(root/'config.json','cpu','fp32')


if __name__=='__main__':unittest.main()
