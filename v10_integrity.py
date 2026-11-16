"""Exact completed-run and frozen-input guards for Age experiments."""
import hashlib
import json
import math
from pathlib import Path
import torch
from clinical import fit_age_stats
from data import load_rows
from internal_split import partition_rows
from provenance import digest
from v8_integrity import verify_evidence


def verify_reference_probabilities(actual,expected):
    if not expected or len(actual)!=len(expected) or any(not math.isfinite(float(p)) or not 0<=p<=1 for p in list(actual)+list(expected)):
        raise ValueError('Invalid reference probability alignment/values')
    delta=max(abs(a-b) for a,b in zip(actual,expected))
    if delta>1e-7:raise ValueError('Original per-image reference probabilities did not reproduce')
    return delta


def recipe(name,full=False,epoch=0):
    return dict(run=name,data='data/raw/glaucoma_first',cache='data/cache384',
        pretrain='pretrain_model/convnext_tiny-983f1562.pth',split_manifest=None if full else 'reports/v5_internal_split.json',
        full=full,resume=False,epochs=8,checkpoint_epoch=epoch,seed=20261041 if full else 20261031,
        batch=16,workers=4,size=384,max_steps=0,architecture='convnext_tiny_age',augmentation='mild',
        memory_format='contiguous',validation_probability_precision='fp32',backbone_lr=.0001,age_lr=.001,
        backbone_weight_decay=.0001,age_weight_decay=.01,pos_weight=1.,clip_grad=5.)


def validate_completed_age(directory,expected,root):
    directory,root=Path(directory),Path(root)
    status=json.loads((directory/'status.json').read_text(encoding='utf-8'))
    if status['state']!='complete':raise ValueError('Age run not complete')
    config=json.loads((directory/'config.json').read_text(encoding='utf-8'))
    for key,value in expected.items():
        if config.get(key)!=value:raise ValueError('Age completed recipe mismatch: '+key)
    saved=torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
    if saved.get('schema_version')!=2 or saved.get('architecture')!='convnext_tiny_age':
        raise ValueError('Completed Age model schema mismatch')
    for key in ('size','memory_format','validation_probability_precision','checkpoint_epoch'):
        if saved.get(key)!=expected[key]:raise ValueError('Completed Age checkpoint mismatch: '+key)
    raw=root/expected['data'];rows=load_rows(raw,'train');split_hash=None
    if expected['split_manifest']:
        rows,_,split_hash=partition_rows(rows,(raw/'train.csv').read_bytes(),root/expected['split_manifest'])
    fitted=fit_age_stats(rows)
    if saved['age_stats']!=fitted or json.loads((directory/'age_stats.json').read_text(encoding='utf-8'))!=fitted:
        raise ValueError('Completed Age transform does not match actual training rows')
    fingerprint=json.loads((root/expected['cache']/'READY.json').read_text(encoding='utf-8'))['manifest_sha256']
    if split_hash:fingerprint=hashlib.sha256((fingerprint+'|'+split_hash).encode()).hexdigest()
    if saved['data_fingerprint']!=fingerprint or saved['split_manifest_sha256']!=split_hash:
        raise ValueError('Completed Age data/split fingerprint mismatch')
    if saved['validation_partition']!=('official_valid' if expected['full'] else 'internal_train_holdout'):
        raise ValueError('Completed Age validation scope mismatch')
    if digest(root/expected['pretrain'])!=saved['public_pretrain_sha256']:
        raise ValueError('Completed Age public-source fingerprint mismatch')
    epoch=saved['metrics']['epoch']
    if type(epoch) is not int or not 1<=epoch<=expected['epochs'] or (expected['checkpoint_epoch'] and epoch!=expected['checkpoint_epoch']):
        raise ValueError('Completed Age checkpoint is not the frozen selected epoch')
    if saved['metrics']!=status['best'] or saved['metrics']!=json.loads((directory/'best_metrics.json').read_text(encoding='utf-8')):
        raise ValueError('Completed Age metrics binding mismatch')
    return dict(config=config,checkpoint=saved)


def frozen(root):
    record=json.loads((Path(root)/'output/v10/frozen_inputs.json').read_text(encoding='utf-8'))
    verify_evidence(record['evidence'])
    return record
