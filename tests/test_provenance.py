import json
import tempfile
import unittest
from pathlib import Path
from provenance import digest,inference_fingerprint,verify_prediction_binding,copy_evidence_alias


class BindingTests(unittest.TestCase):
    def test_evidence_alias_preserves_source_hash_and_existing_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'candidate.json';source.write_bytes(b'original failure')
            target=root/'v7_unfused_candidate.json';expected=digest(source)
            record=copy_evidence_alias(source,target,expected)
            self.assertEqual(target.read_bytes(),source.read_bytes());self.assertEqual(record['sha256'],expected)
            copy_evidence_alias(source,target,expected)
            target.write_bytes(b'preserved different copy')
            with self.assertRaises(ValueError):copy_evidence_alias(source,target,expected)
            self.assertEqual(target.read_bytes(),b'preserved different copy')

    def test_evidence_alias_rejects_changed_source_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.write_bytes(b'original');expected=digest(source)
            source.write_bytes(b'changed');target=root/'target'
            with self.assertRaises(ValueError):copy_evidence_alias(source,target,expected)
            self.assertFalse(target.exists())

    def test_binding_detects_stale_output_weights_and_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            weight=root/'weight.pt'; weight.write_bytes(b'original')
            config=root/'config.json'; config.write_text(json.dumps(dict(models=[dict(checkpoint='weight.pt')])))
            output=root/'submission.csv'; output.write_text('original output')
            output.with_suffix('.manifest.json').write_text(json.dumps(dict(files=inference_fingerprint(config),submission_sha256=digest(output),precision='fp16')))
            verify_prediction_binding(output,config)
            for path in [weight,config,output]:
                original=path.read_bytes(); path.write_bytes(original+b' ')
                with self.assertRaises(ValueError): verify_prediction_binding(output,config)
                path.write_bytes(original)
