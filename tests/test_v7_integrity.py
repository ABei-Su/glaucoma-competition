import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import torch
from v7_integrity import requested_recipe,validate_completed_run


class V7IntegrityTests(unittest.TestCase):
    def prepare(self,root,internal=False):
        recipe=requested_recipe('runs/example',20261033 if internal else 20261034,16,internal,0 if internal else 8)
        directory=root/'runs/example';directory.mkdir(parents=True)
        cache=root/'data/cache384';cache.mkdir(parents=True)
        (cache/'READY.json').write_text(json.dumps(dict(manifest_sha256='cache-sha')))
        fingerprint='cache-sha';split_fingerprint=None
        if internal:
            split=root/'reports/v5_internal_split.json';split.parent.mkdir();split.write_text('{}')
            split_fingerprint=hashlib.sha256(split.read_bytes()).hexdigest()
            fingerprint=hashlib.sha256((fingerprint+'|'+split_fingerprint).encode()).hexdigest()
        metric=dict(epoch=8,auroc=.9,average_precision=.7)
        (directory/'config.json').write_text(json.dumps(recipe))
        (directory/'status.json').write_text(json.dumps(dict(state='complete',best=metric)))
        (directory/'best_metrics.json').write_text(json.dumps(metric))
        torch.save(dict(model={},architecture='efficientnet_v2_s',size=384,memory_format='contiguous',
                        validation_probability_precision='fp32',checkpoint_epoch=recipe['checkpoint_epoch'],
                        data_fingerprint=fingerprint,split_manifest_sha256=split_fingerprint,
                        validation_partition='internal_train_holdout' if internal else 'official_valid',metrics=metric),directory/'best.pt')
        return directory,recipe

    def test_completed_recipe_and_selected_epoch_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);directory,recipe=self.prepare(root)
            validate_completed_run(directory,recipe,root)
            changed=dict(recipe,batch=8)
            with self.assertRaisesRegex(ValueError,'batch'):validate_completed_run(directory,changed,root)
            saved=torch.load(directory/'best.pt',weights_only=True);saved['metrics']['epoch']=7;torch.save(saved,directory/'best.pt')
            with self.assertRaises(ValueError):validate_completed_run(directory,recipe,root)

    def test_completed_internal_split_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);directory,recipe=self.prepare(root,True)
            validate_completed_run(directory,recipe,root)
            (root/'reports/v5_internal_split.json').write_text('{ }')
            with self.assertRaisesRegex(ValueError,'fingerprint'):validate_completed_run(directory,recipe,root)
