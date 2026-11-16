"""Exercise the real Age training loop across an epoch-boundary interruption."""
import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from PIL import Image
from clinical import AgeConvNeXt
from internal_split import make_manifest
import train_age


def tiny_model(*args,**kwargs):
    return AgeConvNeXt(torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),torch.nn.Linear(3,1)))


class AgeTrainingTests(unittest.TestCase):
    def fixture(self,root):
        raw=root/'raw';(raw/'train').mkdir(parents=True)
        rows=[dict(Image_name=f'train_{i}.png',Age=str(20+int(i%2)*60),True_Label=str(i%2)) for i in range(40)]
        with (raw/'train.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['Image_name','Age','True_Label']);writer.writeheader();writer.writerows(rows)
        for row in rows:Image.new('RGB',(32,32),(30+int(row['True_Label'])*170,60,40)).save(raw/'train'/row['Image_name'])
        split=root/'split.json';split.write_text(json.dumps(make_manifest(rows,(raw/'train.csv').read_bytes())))
        cache=root/'cache';cache.mkdir();(cache/'READY.json').write_text('{}')
        public=root/'public.pth';public.write_bytes(b'synthetic public source identity')
        return raw,split,cache,public

    def invoke(self,paths,run,resume=False,interrupt=False):
        raw,split,cache,public=paths
        args=['--data',str(raw),'--cache',str(cache),'--pretrain',str(public),'--split-manifest',str(split),
              '--run',str(run),'--epochs','2','--batch','8','--workers','0','--size','32']+(['--resume'] if resume else [])
        original=train_age.save_json
        def save(path,value):
            original(path,value)
            if interrupt and Path(path).name=='best_metrics.json':raise KeyboardInterrupt('test epoch interruption')
        with patch.object(train_age,'build_age_model',tiny_model),patch.object(train_age,'audit_inputs',return_value={'manifest_sha256':'synthetic'}),\
             patch.object(train_age.torch.cuda,'is_available',return_value=False),patch.object(train_age,'save_json',side_effect=save),\
             contextlib.redirect_stdout(io.StringIO()):train_age.main(args)

    def test_resume_matches_full_and_transform_fits_actual_training_partition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);paths=self.fixture(root)
            self.invoke(paths,root/'full')
            with self.assertRaises(KeyboardInterrupt):self.invoke(paths,root/'resumed',interrupt=True)
            self.assertFalse((root/'resumed/training.lock').exists())
            self.invoke(paths,root/'resumed',resume=True)
            a=torch.load(root/'full/last.pt',weights_only=False);b=torch.load(root/'resumed/last.pt',weights_only=False)
            self.assertEqual(a['age_stats'],b['age_stats']);self.assertEqual(a['age_stats']['training_rows'],34)
            self.assertEqual(a['epoch'],1)
            for name,value in a['model'].items():torch.testing.assert_close(value,b['model'][name],rtol=0,atol=0)
            saved=torch.load(root/'resumed/best.pt',weights_only=True)
            self.assertEqual(saved['architecture'],'convnext_tiny_age');self.assertEqual(saved['schema_version'],2)
            self.assertEqual(saved['validation_partition'],'internal_train_holdout')
            # No official valid.csv or images exist in the fixture.

    def test_resume_rejects_tampered_age_transform_and_full_requires_fixed_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);paths=self.fixture(root);run=root/'run'
            with self.assertRaises(KeyboardInterrupt):self.invoke(paths,run,interrupt=True)
            saved=torch.load(run/'last.pt',weights_only=False);saved['age_stats']['mean']+=1
            torch.save(saved,run/'last.pt')
            with self.assertRaisesRegex(ValueError,'Age transform'):self.invoke(paths,run,resume=True)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):train_age.parse_args(['--run','unused','--full'])
            with self.assertRaises(SystemExit):train_age.parse_args(['--run','unused','--split-manifest',''])


if __name__=='__main__':unittest.main()
