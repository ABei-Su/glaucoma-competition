"""Exercise exact image/Age inference, metadata joins and immutable bindings."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from PIL import Image
from clinical import AgeConvNeXt,fit_age_stats
from clinical_predict import AgeLookup,Predictor,fingerprint,write_age_metadata
from provenance import digest


def tiny_model():
    base=torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),torch.nn.Linear(3,1,bias=False))
    with torch.no_grad():base[-1].weight.zero_()
    return AgeConvNeXt(base)


class ClinicalPredictTests(unittest.TestCase):
    def test_metadata_contains_no_labels_and_handles_unknown_ages_and_collisions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'official.csv'
            source.write_text('Image_name,Age,True_Label\na.JPG,20,0\nb.png,,1\n',encoding='utf-8')
            target=root/'age.csv';write_age_metadata(source,target)
            self.assertEqual(target.read_text(encoding='utf-8').splitlines()[0],'Image_name,Age')
            lookup=AgeLookup(target);stats=fit_age_stats([{'Age':'20'},{'Age':'80'}])
            self.assertEqual(lookup.features('A.jpg',stats).tolist(),[-1.,0.])
            self.assertEqual(lookup.features('unknown.jpg',stats).tolist(),[0.,1.])
            self.assertEqual(lookup.features('b.png',stats).tolist(),[0.,1.])
            with self.assertRaises(FileExistsError):write_age_metadata(source,target)
            with self.assertRaises(ValueError):AgeLookup(source)
            target.write_text('Image_name,Age\na.JPG,20\na.jpg,80\n',encoding='utf-8')
            with self.assertRaises(ValueError):AgeLookup(target)

    def test_new_checkpoint_tta_uses_same_age_and_metadata_is_hash_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);image=root/'sample.png';Image.new('RGB',(32,32),(80,80,80)).save(image)
            stats=fit_age_stats([{'Age':'20'},{'Age':'80'}]);model=tiny_model()
            with torch.no_grad():model.age.weight.copy_(torch.tensor([[2.,3.]]))
            checkpoint=root/'age.pt';torch.save(dict(schema_version=2,architecture='convnext_tiny_age',
                model=model.state_dict(),size=32,age_stats=stats,memory_format='contiguous'),checkpoint)
            metadata=root/'age.csv';metadata.write_text('Image_name,Age\nsample.png,80\n',encoding='utf-8')
            config=root/'config.json';config.write_text(json.dumps(dict(schema_version=2,age_metadata='age.csv',
                age_metadata_sha256=digest(metadata),models=[dict(checkpoint='age.pt',sha256=digest(checkpoint),tta=True)])))
            with patch('clinical_predict.build_age_model',side_effect=lambda *a,**kw:tiny_model()):
                predictor=Predictor(config,device='cpu',precision='fp32')
                self.assertAlmostEqual(predictor.predict_one(image),torch.tensor(2.).sigmoid().item(),places=7)
            before=fingerprint(config);metadata.write_text('Image_name,Age\nsample.png,20\n',encoding='utf-8')
            self.assertNotEqual(before,fingerprint(config))
            with patch('clinical_predict.build_age_model',side_effect=lambda *a,**kw:tiny_model()):
                with self.assertRaisesRegex(ValueError,'metadata.*hash'):Predictor(config,device='cpu',precision='fp32')
            metadata.write_text('Image_name,Age\nsample.png,80\n',encoding='utf-8')
            checkpoint.write_bytes(checkpoint.read_bytes()+b'extra')
            with patch('clinical_predict.build_age_model',side_effect=lambda *a,**kw:tiny_model()):
                with self.assertRaisesRegex(ValueError,'checkpoint.*hash'):Predictor(config,device='cpu',precision='fp32')
            config.write_text(json.dumps(dict(schema_version=2,models=[dict(checkpoint='age.pt',tta=True)])))
            with self.assertRaises(ValueError):Predictor(config,device='cpu',precision='fp32')


if __name__=='__main__':unittest.main()
