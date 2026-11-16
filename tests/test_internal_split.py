import copy
import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
import train

from internal_split import make_manifest, partition_rows


class InternalSplitTests(unittest.TestCase):
    def setUp(self):
        self.rows=[{'Image_name':f'train_{i:03}.jpg','True_Label':str(i%2)} for i in range(40)]
        self.csv=b'synthetic official training labels'

    def test_deterministic_disjoint_complete_and_order_independent(self):
        a=make_manifest(self.rows,self.csv)
        b=make_manifest(list(reversed(self.rows)),self.csv)
        self.assertEqual(a,b)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'split.json';path.write_text(json.dumps(a))
            train,holdout,fingerprint=partition_rows(self.rows,self.csv,path)
        self.assertEqual(len(train),34)
        self.assertEqual(len(holdout),6)
        self.assertFalse({r['Image_name'] for r in train}&{r['Image_name'] for r in holdout})
        self.assertEqual({r['Image_name'] for r in train+holdout},{r['Image_name'] for r in self.rows})
        self.assertEqual(len(fingerprint),64)

    def test_reject_overlap_unknown_missing_or_modified_source(self):
        base=make_manifest(self.rows,self.csv)
        variants=[]
        overlap=copy.deepcopy(base);overlap['train_names'].append(overlap['holdout_names'][0]);variants.append(overlap)
        missing=copy.deepcopy(base);missing['train_names'].pop();variants.append(missing)
        external=copy.deepcopy(base);external['holdout_names'][0]='valid_00000.jpg';variants.append(external)
        changed=copy.deepcopy(base);changed['train_csv_sha256']='0'*64;variants.append(changed)
        duplicate=copy.deepcopy(base);duplicate['train_names'].append(duplicate['train_names'][0]);variants.append(duplicate)
        for manifest in variants:
            with self.subTest(manifest=manifest),tempfile.TemporaryDirectory() as temporary:
                path=Path(temporary)/'split.json';path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):partition_rows(self.rows,self.csv,path)

    def test_real_training_uses_only_internal_holdout_and_rejects_changed_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);raw=root/'raw';(raw/'train').mkdir(parents=True)
            rows=[{'Image_name':f'train_{i}.png','True_Label':str(i%2)} for i in range(8)]
            with (raw/'train.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['Image_name','True_Label']);writer.writeheader();writer.writerows(rows)
            for row in rows:Image.new('RGB',(32,32),(50+150*int(row['True_Label']),30,20)).save(raw/'train'/row['Image_name'])
            # No valid.csv or valid directory exists: accidental official-validation access fails.
            split=root/'split.json';split.write_text(json.dumps(make_manifest(rows,(raw/'train.csv').read_bytes())))
            cache=root/'cache';cache.mkdir();(cache/'READY.json').write_text('{}')
            run=root/'run'
            args=['train.py','--data',str(raw),'--cache',str(cache),'--run',str(run),'--size','32',
                  '--epochs','2','--batch','2','--workers','0','--split-manifest',str(split),'--checkpoint-epoch','2']
            def tiny_model(*args,**kwargs):
                return torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),torch.nn.Linear(3,1))
            def invoke(resume=False):
                with patch('sys.argv',args+(['--resume'] if resume else [])),patch.object(train,'build_model',tiny_model),\
                     patch.object(train,'audit_inputs',return_value={'manifest_sha256':'synthetic'}),\
                     patch.object(train.torch.cuda,'is_available',return_value=False),contextlib.redirect_stdout(io.StringIO()):train.main()
            invoke()
            binding=(run/'split_binding.json').read_bytes()
            checkpoint=torch.load(run/'best.pt',weights_only=True)
            self.assertEqual(checkpoint['validation_partition'],'internal_train_holdout')
            self.assertEqual(checkpoint['metrics']['epoch'],2)
            with (run/'validation_predictions.csv').open() as stream:
                predicted={row['Image_name'] for row in csv.DictReader(stream)}
            self.assertEqual(predicted,set(json.loads(split.read_text())['holdout_names']))
            split.write_text(split.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'fingerprint'):invoke(True)
            self.assertEqual((run/'split_binding.json').read_bytes(),binding)

