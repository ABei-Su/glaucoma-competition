"""Bind inference outputs to their exact immutable input weights/configuration."""
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy_evidence_alias(source,destination,expected_sha256):
    source,destination=Path(source),Path(destination)
    content=source.read_bytes()
    if hashlib.sha256(content).hexdigest()!=expected_sha256:
        raise ValueError('Original evidence changed before copy')
    if destination.exists() and destination.read_bytes()!=content:
        raise ValueError('Preserve different existing evidence copy')
    if not destination.exists():destination.write_bytes(content)
    if digest(destination)!=expected_sha256:raise ValueError('Evidence copy hash mismatch')
    return dict(source=str(source.resolve()),packaged_report_name=destination.name,sha256=expected_sha256)


def inference_fingerprint(checkpoint):
    path=Path(checkpoint).resolve()
    files=[path]
    if path.suffix.lower()=='.json':
        spec=json.loads(path.read_text(encoding='utf-8'))
        files += [(path.parent/item['checkpoint']).resolve() for item in spec['models']]
    return [{'path':str(file),'sha256':digest(file)} for file in files]


def verify_prediction_binding(output,checkpoint,precision='fp16'):
    output=Path(output)
    record=json.loads(output.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    if record['files']!=inference_fingerprint(checkpoint) or record['submission_sha256']!=digest(output) or record['precision']!=precision:
        raise ValueError('Submission/configuration/weights binding mismatch')
    return record
