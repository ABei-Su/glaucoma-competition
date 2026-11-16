import hashlib
import json
from pathlib import Path, PureWindowsPath
import tempfile
import unittest
import torch
import train_focal
from focal_loss import loss_spec
import v12_protocol as protocol
from tools import finalize_v12


class V12ContractTests(unittest.TestCase):
    def test_trial_delivery_uses_declared_twelfth_version_directory(self):
        self.assertTrue(hasattr(finalize_v12,'delivery_directory'),'Declared version delivery guard is missing')
        self.assertEqual(finalize_v12.delivery_directory(Path('E:/bisai/glaucoma_project')),
                         Path('E:/bisai/第十二版试交_v12'))

    def test_fixed_focal_cli_exactly_matches_internal_and_full_recipes(self):
        for internal,epoch in [(True,0),(False,3)]:
            expected=protocol.recipe(internal,epoch)
            self.assertEqual(vars(train_focal.parse_args(protocol.training_command(internal,epoch)[1:])),expected)
            self.assertEqual(expected['focal_gamma'],2.);self.assertEqual(expected['focal_alpha'],-1.)
            self.assertEqual(expected['size'],384);self.assertEqual(expected['full'],not internal)
            self.assertEqual(expected['seed'],20261031 if internal else 20261032)
            self.assertIn('focal',expected['run'])

    def test_completed_guard_requires_focal_metadata_public_identity_and_fixed_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);expected=protocol.recipe(False,3);run=root/expected['run'];run.mkdir(parents=True)
            cache=root/'data/cache384';cache.mkdir(parents=True);(cache/'READY.json').write_text(json.dumps(dict(manifest_sha256='cache')))
            public=root/expected['pretrain'];public.parent.mkdir();public.write_bytes(b'public fixture')
            (run/'config.json').write_text(json.dumps(expected))
            history=[dict(epoch=i,auroc=.8+i/100,average_precision=.5+i/100) for i in range(1,9)];best=history[2]
            (run/'history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
            (run/'status.json').write_text(json.dumps(dict(state='complete',best=best)))
            (run/'best_metrics.json').write_text(json.dumps(best))
            saved=dict(model={},architecture='convnext_tiny',size=384,memory_format='contiguous',
                validation_probability_precision='fp32',checkpoint_epoch=3,augmentation='mild',data_fingerprint='cache',
                split_manifest_sha256=None,validation_partition='official_valid',metrics=best,loss_spec=loss_spec(),
                public_pretrain_sha256=hashlib.sha256(public.read_bytes()).hexdigest())
            torch.save(saved,run/'best.pt');protocol.validate_completed(root,False,3)
            saved['loss_spec']=loss_spec(gamma=1);torch.save(saved,run/'best.pt')
            with self.assertRaisesRegex(ValueError,'loss contract'):protocol.validate_completed(root,False,3)
            saved['loss_spec']=loss_spec();torch.save(saved,run/'best.pt');public.write_bytes(b'changed public')
            with self.assertRaisesRegex(ValueError,'public-source'):protocol.validate_completed(root,False,3)

    def test_original_and_derived_sources_match_auditable_record(self):
        root=Path(__file__).resolve().parents[1]
        record=json.loads((root/'reports/v12_derivation.json').read_text(encoding='utf-8'))
        self.assertEqual(len(record['files']),5)
        for item in record['files']:
            for role in ('source', 'target'):
                archived_path = PureWindowsPath(item[role])
                project_index = archived_path.parts.index('glaucoma_project')
                checkout_path = root.joinpath(*archived_path.parts[project_index + 1:])
                self.assertEqual(hashlib.sha256(checkout_path.read_bytes()).hexdigest(), item[role + '_sha256'])


if __name__=='__main__':unittest.main()
