import tempfile
import unittest
from pathlib import Path
import torch
from PIL import Image
from model import build_model
from predict import Predictor
from train import parse_args


class ArchitectureTests(unittest.TestCase):
    def test_efficientnet_public_state_load_and_frozen_reload(self):
        from torchvision.models import efficientnet_v2_s
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            original=efficientnet_v2_s(weights=None)
            pretrain=root/'public-shape.pt';torch.save(original.state_dict(),pretrain)
            model=build_model(pretrain,architecture='efficientnet_v2_s').eval()
            self.assertEqual(model.classifier[-1].out_features,1)
            torch.testing.assert_close(model.features[0][0].weight,original.features[0][0].weight)
            image=root/'sample.png';Image.new('RGB',(80,60),(80,110,140)).save(image)
            checkpoint=root/'model.pt'
            torch.save(dict(model=model.state_dict(),size=64,architecture='efficientnet_v2_s'),checkpoint)
            predictor=Predictor(checkpoint,'cpu','fp32')
            self.assertEqual(predictor.predict_one(image),predictor.predict_one(image))
            self.assertTrue(0<=predictor.predict_one(image)<=1)
            self.assertFalse(predictor.model.training)
            self.assertTrue(all(not p.requires_grad for p in predictor.model.parameters()))
        args=parse_args(['--architecture','efficientnet_v2_s'])
        self.assertTrue(args.pretrain.endswith('efficientnet_v2_s-dd5fe13b.pth'))
        self.assertEqual(args.validation_probability_precision,'fp32')

    def test_convnext_binary_head_and_frozen_reload(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            model=build_model(architecture='convnext_tiny').eval()
            self.assertEqual(model.classifier[-1].out_features,1)
            image=root/'sample.png';Image.new('RGB',(80,60),(70,120,50)).save(image)
            checkpoint=root/'model.pt'
            torch.save(dict(model=model.state_dict(),size=64,architecture='convnext_tiny'),checkpoint)
            predictor=Predictor(checkpoint,'cpu','fp32')
            self.assertEqual(predictor.predict_one(image),predictor.predict_one(image))
            self.assertFalse(predictor.model.training)
            self.assertTrue(all(not p.requires_grad for p in predictor.model.parameters()))
        args=parse_args(['--architecture','convnext_tiny'])
        self.assertTrue(args.pretrain.endswith('convnext_tiny-983f1562.pth'))
        self.assertEqual(args.validation_probability_precision,'fp32')

    def test_resnet50_saved_identity_and_binary_head(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            model=build_model(architecture='resnet50').eval()
            self.assertEqual(model.fc.in_features,2048)
            self.assertEqual(model.fc.out_features,1)
            image=root/'sample.png'; Image.new('RGB',(80,60),(120,30,60)).save(image)
            checkpoint=root/'model.pt'
            torch.save(dict(model=model.state_dict(),size=64,architecture='resnet50',memory_format='channels_last'),checkpoint)
            predictor=Predictor(checkpoint,'cpu','fp32')
            self.assertEqual(predictor.model.fc.in_features,2048)
            self.assertTrue(predictor.model.conv1.weight.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(0<=predictor.predict_one(image)<=1)
            self.assertFalse(predictor.model.training)

    def test_architecture_options_and_rejection(self):
        self.assertEqual(parse_args([]).architecture,'resnet18')
        args=parse_args(['--architecture','resnet50'])
        self.assertTrue(args.pretrain.endswith('resnet50-11ad3fa6.pth'))
        self.assertEqual(args.validation_probability_precision,'fp32')
        self.assertEqual(parse_args([]).validation_probability_precision,'fp16')
        with self.assertRaises(ValueError): build_model(architecture='unknown')
