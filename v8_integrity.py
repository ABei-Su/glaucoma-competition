"""Frozen inputs and exact training recipes for the v8 controlled experiment."""
import hashlib
import json
from pathlib import Path
from v7_integrity import requested_recipe,validate_completed_run


def recipe(name,seed,internal=False,epoch=0,augmentation='color_robust'):
    result=requested_recipe('runs/'+name,seed,16,internal,epoch)
    result.update(architecture='convnext_tiny',pretrain='pretrain_model/convnext_tiny-983f1562.pth',augmentation=augmentation)
    return result


def control(root):
    expected=recipe('v5_inner_convnext',20261031,True,augmentation='mild');expected.pop('augmentation')
    checked=validate_completed_run(Path(root)/'runs/v5_inner_convnext',expected,root)
    if checked['config'].get('augmentation','mild')!='mild' or checked['checkpoint'].get('augmentation','mild')!='mild':
        raise ValueError('Original control is not the mild recipe')
    return checked


def verify_evidence(items):
    for item in items:
        if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Frozen v8 evidence changed: '+item['path'])


def frozen(root):
    record=json.loads((Path(root)/'output/v8/frozen_inputs.json').read_text())
    verify_evidence(record['evidence'])
    return record
