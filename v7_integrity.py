"""Verify a completed run against the exact frozen recipe before reuse."""
import hashlib
import json
from pathlib import Path
import torch


def requested_recipe(name,seed,batch,internal=False,epoch=0):
    return dict(data='data/raw/glaucoma_first',cache='data/cache384',run=name,
                architecture='efficientnet_v2_s',pretrain='pretrain_model/efficientnet_v2_s-dd5fe13b.pth',
                size=384,epochs=8,lr=.0001,batch=batch,pos_weight=1.,seed=seed,workers=4,max_steps=0,
                init_checkpoint=None,memory_format='contiguous',validation_probability_precision='fp32',
                split_manifest='reports/v5_internal_split.json' if internal else None,checkpoint_epoch=epoch)


def validate_completed_run(directory,expected,root):
    directory,root=Path(directory),Path(root)
    status=json.loads((directory/'status.json').read_text())
    if status['state']!='complete':raise ValueError('Run not complete')
    config=json.loads((directory/'config.json').read_text())
    for key,value in expected.items():
        if config.get(key)!=value:raise ValueError('Completed recipe mismatch: '+key)
    saved=torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
    if 'augmentation' in expected and saved.get('augmentation','mild')!=expected['augmentation']:
        raise ValueError('Completed checkpoint mismatch: augmentation')
    for key in ['architecture','size','memory_format','validation_probability_precision','checkpoint_epoch']:
        if saved.get(key)!=expected[key]:raise ValueError('Completed checkpoint mismatch: '+key)
    fingerprint=json.loads((root/expected['cache']/'READY.json').read_text())['manifest_sha256']
    split_fingerprint=None
    if expected['split_manifest']:
        split_fingerprint=hashlib.sha256((root/expected['split_manifest']).read_bytes()).hexdigest()
        fingerprint=hashlib.sha256((fingerprint+'|'+split_fingerprint).encode()).hexdigest()
    if saved['data_fingerprint']!=fingerprint or saved.get('split_manifest_sha256')!=split_fingerprint:
        raise ValueError('Completed data/split fingerprint mismatch')
    partition='internal_train_holdout' if expected['split_manifest'] else 'official_valid'
    if saved.get('validation_partition')!=partition:raise ValueError('Completed validation partition mismatch')
    epoch=saved['metrics']['epoch']
    if type(epoch) is not int or not 1<=epoch<=expected['epochs'] or (expected['checkpoint_epoch'] and epoch!=expected['checkpoint_epoch']):
        raise ValueError('Completed checkpoint is not the selected epoch')
    if saved['metrics']!=status['best'] or saved['metrics']!=json.loads((directory/'best_metrics.json').read_text()):
        raise ValueError('Completed metric/checkpoint binding mismatch')
    return dict(config=config,checkpoint=saved)
