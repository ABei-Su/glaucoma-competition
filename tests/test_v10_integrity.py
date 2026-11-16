"""Reject mismatched Age run recipes, fitted transforms and selected epochs."""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import torch
from clinical import fit_age_stats
from internal_split import make_manifest,partition_rows
from v10_integrity import recipe,validate_completed_age,verify_reference_probabilities


class V10IntegrityTests(unittest.TestCase):
    def test_reference_reproduction_rejects_rank_preserving_probability_changes(self):
        self.assertEqual(verify_reference_probabilities([.1,.3,.7],[.1,.3,.7]),0.)
        self.assertLess(verify_reference_probabilities([.1,.3,.70000001],[.1,.3,.7]),1e-7)
        for actual in ([.2,.4,.8],[.1,.3],[.1,float('nan'),.7]):
            with self.assertRaises(ValueError):verify_reference_probabilities(actual,[.1,.3,.7])

    def fixture(self,root,full=False):
        expected=recipe('runs/example',full,8 if full else 0)
        directory=root/'runs/example';directory.mkdir(parents=True)
        raw=root/'data/raw/glaucoma_first';raw.mkdir(parents=True)
        rows=[dict(Image_name=f'train_{i}.png',Age=str(20+i),True_Label=str(i%2)) for i in range(40)]
        with (raw/'train.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['Image_name','Age','True_Label']);writer.writeheader();writer.writerows(rows)
        cache=root/'data/cache384';cache.mkdir(parents=True);(cache/'READY.json').write_text('{"manifest_sha256":"cache-sha"}')
        public=root/expected['pretrain'];public.parent.mkdir();public.write_bytes(b'public-source')
        split_hash=None
        if not full:
            split=root/expected['split_manifest'];split.parent.mkdir();split.write_text(json.dumps(make_manifest(rows,(raw/'train.csv').read_bytes())))
            rows,_,split_hash=partition_rows(rows,(raw/'train.csv').read_bytes(),split)
        fingerprint='cache-sha' if full else hashlib.sha256(('cache-sha|'+split_hash).encode()).hexdigest()
        age=fit_age_stats(rows);metric=dict(epoch=8,auroc=.99,average_precision=.85)
        for name,value in [('config.json',expected),('status.json',dict(state='complete',best=metric)),
                           ('best_metrics.json',metric),('age_stats.json',age)]:
            (directory/name).write_text(json.dumps(value))
        torch.save(dict(schema_version=2,architecture='convnext_tiny_age',model={},size=384,memory_format='contiguous',
            validation_probability_precision='fp32',checkpoint_epoch=expected['checkpoint_epoch'],
            data_fingerprint=fingerprint,split_manifest_sha256=split_hash,
            validation_partition='official_valid' if full else 'internal_train_holdout',metrics=metric,age_stats=age,
            public_pretrain_sha256=hashlib.sha256(public.read_bytes()).hexdigest()),directory/'best.pt')
        return directory,expected

    def test_completed_age_recipe_and_transform_must_match_training_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);directory,expected=self.fixture(root)
            validate_completed_age(directory,expected,root)
            with self.assertRaisesRegex(ValueError,'seed'):validate_completed_age(directory,dict(expected,seed=1),root)
            age=json.loads((directory/'age_stats.json').read_text());age['mean']+=1
            (directory/'age_stats.json').write_text(json.dumps(age))
            with self.assertRaisesRegex(ValueError,'Age transform'):validate_completed_age(directory,expected,root)

    def test_full_run_cannot_substitute_an_official_valid_selected_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);directory,expected=self.fixture(root,True)
            validate_completed_age(directory,expected,root)
            saved=torch.load(directory/'best.pt',weights_only=True);saved['metrics']['epoch']=7;torch.save(saved,directory/'best.pt')
            with self.assertRaises(ValueError):validate_completed_age(directory,expected,root)


if __name__=='__main__':unittest.main()
