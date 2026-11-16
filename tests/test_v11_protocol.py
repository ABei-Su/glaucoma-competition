import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import torch
from train import parse_args
try:
    import v11_protocol as protocol
except ModuleNotFoundError:
    protocol=None


class V11ProtocolTests(unittest.TestCase):
    def ready(self):
        self.assertIsNotNone(protocol,'v11 protocol is not implemented')

    def test_cli_exactly_reproduces_fixed_scope_seed_resolution_and_epoch(self):
        self.ready()
        for internal,epoch in [(True,0),(False,3)]:
            expected=protocol.recipe(internal,epoch)
            self.assertEqual(vars(parse_args(protocol.training_command(internal,epoch)[1:])),expected)
            self.assertEqual(expected['size'],448);self.assertEqual(expected['batch'],16)
            self.assertIsNone(expected['init_checkpoint'])
        for internal,epoch in [(False,0),(False,9),(False,True),(True,3)]:
            with self.assertRaises(ValueError):protocol.recipe(internal,epoch)

    def test_resource_and_quality_gates_reject_nonfinite_and_point_regression(self):
        self.ready()
        self.assertTrue(protocol.resource_gate(4_900_000_000,100.,0.,200_000_000))
        for peak,ms,delta,size in [(5_000_000_000,90.,0.,200_000_000),(1,100.01,0.,1),
                                  (1,float('nan'),0.,1),(1,90.,float('nan'),1),(1,90.,0.,500_000_001)]:
            self.assertFalse(protocol.resource_gate(peak,ms,delta,size))
        baseline=dict(auroc=.98,average_precision=.8)
        self.assertTrue(protocol.quality_gate(baseline,dict(auroc=.981,average_precision=.8))['passed'])
        for scores in [dict(auroc=.98,average_precision=.81),dict(auroc=.981,average_precision=.799),
                       dict(auroc=float('nan'),average_precision=.81)]:
            self.assertFalse(protocol.quality_gate(baseline,scores)['passed'])

    def test_state_guards_reject_unqualified_full_changed_epoch_and_terminal_retry(self):
        self.ready();state=dict(kind='experiment',phase='ready',revision=0,history=[])
        original=copy.deepcopy(state);state=protocol.transition(state,'preflight',{})
        with self.assertRaises(ValueError):protocol.transition(state,'internal_training',{})
        state=protocol.transition(state,'internal_training',dict(resource_passed=True))
        state=protocol.transition(state,'internal_comparison',{})
        with self.assertRaises(ValueError):protocol.transition(state,'full_training',dict(gate={'passed':False},frozen_epoch=3))
        state=protocol.transition(state,'full_training',dict(gate={'passed':True},frozen_epoch=3))
        with self.assertRaises(ValueError):protocol.transition(state,'full_ready',dict(frozen_epoch=4))
        state=protocol.transition(state,'full_ready',dict(frozen_epoch=3))
        with self.assertRaises(ValueError):protocol.transition(state,'preflight',{})
        self.assertEqual(original,dict(kind='experiment',phase='ready',revision=0,history=[]))
        with self.assertRaises(ValueError):protocol.transition(original,'preflight',dict(phase='complete'))

    def test_finalization_rejects_unfrozen_prediction_and_partial_completion(self):
        self.ready();state=dict(kind='finalization',phase='ready',revision=0,history=[])
        state=protocol.transition(state,'waiting_for_full',{})
        with self.assertRaises(ValueError):protocol.transition(state,'timing',{})
        state=protocol.transition(state,'timing',dict(full_ready=True))
        state=protocol.transition(state,'validating',{})
        with self.assertRaises(ValueError):protocol.transition(state,'predicting',dict(resource_passed=True))
        state=protocol.transition(state,'predicting',dict(resource_passed=True,trial_frozen=True))
        state=protocol.transition(state,'checking',{})
        with self.assertRaises(ValueError):protocol.transition(state,'complete',dict(rows=5855))
        state=protocol.transition(state,'complete',dict(rows=5855,delivery_verified=True))
        with self.assertRaises(ValueError):protocol.transition(state,'predicting',{})

    def test_pinned_inference_rejects_changed_member_or_invalid_shape_contract(self):
        self.ready()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);weight=root/'weights.pt';weight.write_bytes(b'fixed test bytes')
            member=dict(checkpoint='weights.pt',sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),tta=True)
            config=root/'config.json';spec=dict(schema_version=1,models=[member,member],batch=1,
                precision='fp16',probability_precision='fp32',threshold=.5)
            config.write_text(json.dumps(spec));identity=protocol.inference_identity(config)
            self.assertTrue(identity)
            weight.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash'):protocol.inference_identity(config)
            weight.write_bytes(b'fixed test bytes');spec['models'][0]['tta']=False;config.write_text(json.dumps(spec))
            with self.assertRaises(ValueError):protocol.inference_identity(config)

    def test_completed_guard_rejects_short_history_wrong_scope_and_changed_cache(self):
        self.ready()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'runs/v11_full_448';run.mkdir(parents=True)
            cache=root/'data/cache448';cache.mkdir(parents=True)
            (cache/'READY.json').write_text(json.dumps(dict(manifest_sha256='frozen-cache')))
            config=protocol.recipe(False,3);(run/'config.json').write_text(json.dumps(config))
            history=[dict(epoch=i,auroc=.8+i/100,average_precision=.5+i/100) for i in range(1,9)]
            best=history[2]
            (run/'history.jsonl').write_text(''.join(json.dumps(item)+'\n' for item in history))
            (run/'status.json').write_text(json.dumps(dict(state='complete',best=best)))
            (run/'best_metrics.json').write_text(json.dumps(best))
            saved=dict(model={},architecture='convnext_tiny',size=448,memory_format='contiguous',
                validation_probability_precision='fp32',checkpoint_epoch=3,augmentation='mild',
                data_fingerprint='frozen-cache',split_manifest_sha256=None,validation_partition='official_valid',metrics=best)
            torch.save(saved,run/'best.pt');protocol.validate_completed(root,False,3)
            (run/'history.jsonl').write_text(''.join(json.dumps(item)+'\n' for item in history[:3]))
            with self.assertRaisesRegex(ValueError,'history'):protocol.validate_completed(root,False,3)
            (run/'history.jsonl').write_text(''.join(json.dumps(item)+'\n' for item in history))
            saved['validation_partition']='internal_train_holdout';torch.save(saved,run/'best.pt')
            with self.assertRaises(ValueError):protocol.validate_completed(root,False,3)
            saved['validation_partition']='official_valid';torch.save(saved,run/'best.pt')
            (cache/'READY.json').write_text(json.dumps(dict(manifest_sha256='changed')))
            with self.assertRaises(ValueError):protocol.validate_completed(root,False,3)


if __name__=='__main__':unittest.main()
