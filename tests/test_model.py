import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import build_model,probabilities_from_logits
from predict import Predictor
from utils import metrics


class ModelTests(unittest.TestCase):
    def test_fp32_probability_preserves_close_half_logit_ranking(self):
        logits=torch.tensor([7.,7.01],dtype=torch.float16)
        self.assertEqual(logits.sigmoid()[0],logits.sigmoid()[1])
        p=probabilities_from_logits(logits)
        self.assertEqual(p.dtype,torch.float32)
        self.assertLess(p[0],p[1])
        self.assertTrue(torch.all((p>0)&(p<1)))

    def test_frozen_predictor_reload_consistency(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            model=build_model().eval()
            path=root/'model.pt'
            torch.save({'model':model.state_dict(),'size':64},path)
            image=root/'image.png'
            Image.new('RGB',(90,60),(100,60,20)).save(image)
            predictor=Predictor(path,'cpu','fp32')
            a=predictor.predict_one(image)
            b=predictor.predict_one(image)
            self.assertEqual(a,b)
            self.assertTrue(0<=a<=1)
            self.assertFalse(predictor.model.training)
            self.assertTrue(all(not p.requires_grad for p in predictor.model.parameters()))
            self.assertTrue(all(p.grad is None for p in predictor.model.parameters()))

    def test_binary_learning_recovers_synthetic_signal(self):
        torch.manual_seed(8)
        x=torch.cat([torch.full((16,1),-1.),torch.full((16,1),1.)])
        labels=torch.cat([torch.zeros(16),torch.ones(16)])
        model=torch.nn.Linear(1,1)
        optimizer=torch.optim.SGD(model.parameters(),lr=.5)
        for _ in range(20):
            optimizer.zero_grad()
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x).flatten(),labels)
            loss.backward()
            optimizer.step()
        p=model(x).flatten().sigmoid().detach().numpy()
        self.assertGreater(metrics(labels.numpy(),p)['auroc'],.99)
        self.assertLess(loss.item(),.2)


if __name__=='__main__':
    unittest.main()
