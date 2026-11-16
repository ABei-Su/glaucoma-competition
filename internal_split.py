"""Auditable partitions of official TRAIN only; never expose official valid/test."""
import hashlib
import json
from pathlib import Path


def make_manifest(rows, csv_bytes, seed=20261029, fraction=.15):
    if not isinstance(seed,int) or isinstance(seed,bool) or not 0<fraction<1:
        raise ValueError('Invalid split seed or fraction')
    names=[row['Image_name'] for row in rows]
    if len(names)!=len(set(names)) or any(row['True_Label'] not in ('0','1') for row in rows):
        raise ValueError('Invalid training rows')
    holdout=[]
    for label in ('0','1'):
        group=[r['Image_name'] for r in rows if r['True_Label']==label]
        if len(group)<2:raise ValueError('Each class needs training and holdout examples')
        group.sort(key=lambda name:(hashlib.sha256((str(seed)+'|'+name).encode()).hexdigest(),name))
        count=max(1,min(len(group)-1,round(len(group)*fraction)))
        holdout.extend(group[:count])
    held=set(holdout)
    return dict(schema_version=1,source_split='train',seed=seed,fraction=fraction,
                method='class-stratified SHA256(seed|Image_name) order',
                train_csv_sha256=hashlib.sha256(csv_bytes).hexdigest(),
                train_names=sorted(set(names)-held),holdout_names=sorted(held))


def partition_rows(rows, csv_bytes, manifest_path):
    content=Path(manifest_path).read_bytes()
    manifest=json.loads(content)
    expected=make_manifest(rows,csv_bytes,manifest.get('seed'),manifest.get('fraction',0))
    if manifest!=expected:
        raise ValueError('Internal split overlap, incomplete coverage, unknown names or source mismatch')
    train=set(manifest['train_names']);holdout=set(manifest['holdout_names'])
    return ([r for r in rows if r['Image_name'] in train],
            [r for r in rows if r['Image_name'] in holdout],hashlib.sha256(content).hexdigest())
