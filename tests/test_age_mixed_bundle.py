import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from PIL import Image
from clinical import AgeConvNeXt,fit_age_stats
from clinical_predict import Predictor
from provenance import digest


def zero_image():
    model=torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),torch.nn.Linear(3,1,bias=False))
    with torch.no_grad():model[-1].weight.zero_()
    return model


class AgeMixedBundleTests(unittest.TestCase):
    def test_mixed_image_age_tta_means_probabilities_and_unknown_age_uses_fixed_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);image=root/'sample.png';Image.new('RGB',(80,40),(100,60,30)).save(image)
            metadata=root/'ages.csv';metadata.write_text('Image_name,Age\nsample.png,80\n',encoding='utf-8')
            stats=fit_age_stats([{'Age':'20'},{'Age':'80'}]);age=AgeConvNeXt(zero_image())
            with torch.no_grad():age.age.weight.copy_(torch.tensor([[2.,3.]]))
            paths=[root/'image.pt',root/'age.pt']
            torch.save(dict(architecture='resnet18',model=zero_image().state_dict(),size=32),paths[0])
            torch.save(dict(schema_version=2,architecture='convnext_tiny_age',model=age.state_dict(),size=64,age_stats=stats),paths[1])
            config=root/'config.json';config.write_text(json.dumps(dict(schema_version=2,
                age_metadata=metadata.name,age_metadata_sha256=digest(metadata),
                models=[dict(checkpoint=p.name,sha256=digest(p),tta=True) for p in paths])))
            with patch('clinical_predict.build_model',side_effect=lambda **kw:zero_image()),\
                 patch('clinical_predict.build_age_model',side_effect=lambda:AgeConvNeXt(zero_image())):
                predictor=Predictor(config,'cpu','fp32')
                self.assertAlmostEqual(predictor.predict_one(image),.6903985142707825,places=7)
                unknown=root/'unknown.png';Image.new('RGB',(80,40),(100,60,30)).save(unknown)
                self.assertAlmostEqual(predictor.predict_one(unknown),.7262870669364929,places=7)


if __name__=='__main__':unittest.main()
