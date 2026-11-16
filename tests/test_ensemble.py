import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image,ImageDraw

from model import build_model
from predict import Predictor
from data import preprocess_image


class EnsembleTests(unittest.TestCase):
    def test_two_models_equal_mean_and_horizontal_tta(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            image=root/'image.png'
            asymmetric=Image.new('RGB',(90,70),(130,50,10))
            ImageDraw.Draw(asymmetric).rectangle((5,8,28,49),fill=(15,220,70))
            asymmetric.save(image)
            for seed in [1,2]:
                torch.manual_seed(seed)
                model=build_model().eval()
                torch.save({'model':model.state_dict(),'size':64},root/f'{seed}.pt')
            a=Predictor(root/'1.pt','cpu','fp32').predict_one(image)
            b=Predictor(root/'2.pt','cpu','fp32').predict_one(image)
            spec={'schema_version':1,'models':[{'checkpoint':'1.pt','tta':False},{'checkpoint':'2.pt','tta':False}]}
            config=root/'config.json'
            config.write_text(json.dumps(spec))
            predictor=Predictor(config,'cpu','fp32')
            self.assertAlmostEqual(predictor.predict_one(image),(a+b)/2,places=7)
            spec['models'][0]['tta']=True
            config.write_text(json.dumps(spec))
            predictor=Predictor(config,'cpu','fp32')
            self.assertEqual(predictor.predict_one(image),predictor.predict_one(image))
            tensor=predictor.transform(preprocess_image(asymmetric,64)).unsqueeze(0)
            with torch.inference_mode():
                first=predictor.members[0][0]
                explicit=(first(tensor).sigmoid().item()+first(tensor.flip(-1)).sigmoid().item())/2
            self.assertAlmostEqual(predictor.predict_one(image),(explicit+b)/2,places=7)

    def test_invalid_bundle_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config=Path(directory)/'config.json'
            config.write_text(json.dumps({'schema_version':1,'models':[]}))
            with self.assertRaises(ValueError):
                Predictor(config,'cpu','fp32')


if __name__=='__main__':
    unittest.main()
