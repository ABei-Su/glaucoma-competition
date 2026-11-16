"""Isolated v13 split and soft-target contracts; no filesystem or GPU effects."""
import copy
import hashlib
import json
import math
import re

import numpy as np
import torch
from torch.nn import functional as F


def mix_batch(images, labels, lam, permutation):
    n = len(images)
    if (not math.isfinite(lam) or not 0 <= lam <= 1 or len(labels) != n
            or permutation.dtype != torch.int64 or permutation.ndim != 1
            or sorted(permutation.cpu().tolist()) != list(range(n))):
        raise ValueError('Invalid Mixup batch, lambda or permutation')
    permutation = permutation.to(images.device)
    return (lam*images+(1-lam)*images[permutation],
            lam*labels+(1-lam)*labels[permutation])


class Mixup:
    def __init__(self, seed):
        self.beta = np.random.default_rng(seed)
        self.permutation = torch.Generator().manual_seed(seed)

    def __call__(self, images, labels):
        return mix_batch(images, labels, float(self.beta.beta(.2, .2)),
                         torch.randperm(len(images), generator=self.permutation))

    def state_dict(self):
        return {'beta': copy.deepcopy(self.beta.bit_generator.state),
                'permutation': self.permutation.get_state().clone()}

    def load_state_dict(self, state):
        self.beta.bit_generator.state = copy.deepcopy(state['beta'])
        self.permutation.set_state(state['permutation'])


def soft_bce(logits, labels):
    # Explicit FP32 loss also inside an autocast forward context.
    with torch.autocast(device_type=logits.device.type, enabled=False):
        return F.binary_cross_entropy_with_logits(logits.float(), labels.float())


def validate_folds(rows, csv_bytes, image_bytes, manifest, fold):
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    sha = lambda b: hashlib.sha256(b).hexdigest()
    require(type(fold) is int and fold in range(3), 'Invalid fold index')
    require(manifest.get('schema_version') == 2 and manifest.get('source_split') == 'train'
            and manifest.get('seed') == 20261029
            and manifest.get('method') == 'StratifiedGroupKFold exact byte hash groups'
            and manifest.get('patient_independence_verified') is False, 'Invalid split contract')
    require(manifest.get('train_csv_sha256') == sha(csv_bytes), 'Train CSV identity mismatch')
    require(manifest.get('image_manifest_sha256') == sha(image_bytes), 'Image manifest identity mismatch')
    names = [r['Image_name'] for r in rows]
    require(len(set(names)) == len(names) and bool(names), 'Empty or duplicate CSV names')
    require(all(r['True_Label'] in ('0', '1') for r in rows), 'Invalid labels')
    require(all(n and '/' not in n and '\\' not in n and ':' not in n for n in names), 'Invalid filename')
    labels = {r['Image_name']: int(r['True_Label']) for r in rows}
    items = [i for i in json.loads(image_bytes) if i['split'] == 'train']
    groups = {i['name']: i['sha256'] for i in items}
    require(len(groups) == len(items) and set(groups) == set(names), 'Image names mismatch')
    require(all(isinstance(h, str) and re.fullmatch('[0-9a-f]{64}', h) for h in groups.values()), 'Invalid image hash')
    group_labels = {}
    for name, group in groups.items():
        require(group not in group_labels or group_labels[group] == labels[name], 'Conflicting labels for exact image group')
        group_labels[group] = labels[name]
    folds = manifest.get('folds', [])
    require(len(folds) == 3, 'Three folds required')
    covered = []
    for index, part in enumerate(folds):
        a, b = part['train_names'], part['holdout_names']
        require(part['fold'] == index and len(set(a)) == len(a) and len(set(b)) == len(b), 'Duplicate or reordered fold')
        require(not set(a) & set(b) and set(a) | set(b) == set(names), 'Fold coverage or overlap')
        require(len(a) == part['train_rows'] and len(b) == part['holdout_rows'], 'Fold counts mismatch')
        require({labels[n] for n in a} == {0, 1} and {labels[n] for n in b} == {0, 1}, 'Empty class')
        require(sum(labels[n] for n in b) == part['holdout_positive'], 'Positive count mismatch')
        require(not {groups[n] for n in a} & {groups[n] for n in b}, 'Exact image group crosses fold')
        covered.extend(b)
    require(len(covered) == len(names) and set(covered) == set(names), 'Holdouts must cover each row exactly once')
    selected = folds[fold]
    by_name = {r['Image_name']: r for r in rows}
    return ([by_name[n] for n in selected['train_names']],
            [by_name[n] for n in selected['holdout_names']])


def promotion(deltas, control_ap, candidate_ap):
    return (len(deltas) == 3 and all(math.isfinite(d) for d in deltas)
            and math.isfinite(control_ap) and math.isfinite(candidate_ap)
            and sum(deltas)/3 >= .0005 and sum(d > 0 for d in deltas) >= 2
            and min(deltas) >= -.001 and candidate_ap >= control_ap)
