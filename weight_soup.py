"""Fixed equal-weight ConvNeXt composition; no fitting or inference changes."""
import math
import torch

CONDITIONS = ('clean', 'dim', 'desaturated')


def average_pair(left, right):
    if not left or set(left) != set(right):
        raise ValueError('Soup requires identical nonempty state keys')
    result = {}
    for name, a in left.items():
        b = right[name]
        if any(part in name for part in ('running_mean', 'running_var', 'num_batches_tracked')):
            raise ValueError('Running-statistic recalibration is outside this experiment')
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise ValueError('Soup state entries must be tensors')
        if a.dtype != torch.float32 or b.dtype != torch.float32 or a.shape != b.shape:
            raise ValueError('Soup requires matching FP32 tensor shapes: ' + name)
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('Nonfinite soup source: ' + name)
        result[name] = (a.detach().cpu().double() * .5 + b.detach().cpu().double() * .5).float()
    return result


def validate_pair(left_config, right_config, left_checkpoint, right_checkpoint):
    keys = ('architecture', 'pretrain', 'seed', 'size', 'data', 'cache', 'split_manifest',
            'epochs', 'batch', 'workers', 'lr', 'pos_weight', 'memory_format',
            'validation_probability_precision', 'init_checkpoint', 'max_steps', 'checkpoint_epoch')
    for key in keys:
        if key not in left_config or key not in right_config or left_config[key] != right_config[key]:
            raise ValueError('Soup training/initialization mismatch: ' + key)
    if left_config['architecture'] != 'convnext_tiny' or left_config['init_checkpoint'] is not None:
        raise ValueError('Only public-initialized ConvNeXt sources are eligible')
    if left_config.get('augmentation', 'mild') != 'mild' or right_config.get('augmentation') != 'color_robust':
        raise ValueError('Soup requires the fixed mild/color-robust source pair')
    keys = ('architecture', 'size', 'memory_format', 'data_fingerprint',
            'split_manifest_sha256', 'validation_partition', 'validation_probability_precision')
    for key in keys:
        if key not in left_checkpoint or key not in right_checkpoint or left_checkpoint[key] != right_checkpoint[key]:
            raise ValueError('Soup checkpoint/data mismatch: ' + key)
    for saved in (left_checkpoint, right_checkpoint):
        if saved.get('metrics', {}).get('epoch') != 8:
            raise ValueError('Soup sources must be the frozen epoch8 checkpoints')
        for key in ('architecture', 'size', 'memory_format', 'validation_probability_precision'):
            if saved[key] != left_config[key]:
                raise ValueError('Soup checkpoint/config mismatch: ' + key)
        internal = left_config['split_manifest'] is not None
        expected = 'internal_train_holdout' if internal else 'official_valid'
        if saved['validation_partition'] != expected or (saved['split_manifest_sha256'] is not None) != internal:
            raise ValueError('Soup sources cross the train/validation boundary')


def soup_gate(reference, candidate):
    for values in (reference, candidate):
        if set(values) != set(CONDITIONS):
            raise ValueError('Exactly the three fixed conditions are required')
        for item in values.values():
            for key in ('auroc', 'average_precision'):
                value = item[key]
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError('Invalid soup metric')
    clean_gain = candidate['clean']['auroc'] - reference['clean']['auroc']
    worst_gain = min(x['auroc'] for x in candidate.values()) - min(x['auroc'] for x in reference.values())
    ap_gain = candidate['clean']['average_precision'] - reference['clean']['average_precision']
    return dict(passed=clean_gain >= .0005 - 1e-12 and worst_gain >= .0005 - 1e-12 and ap_gain >= -1e-12,
                clean_auroc_gain=clean_gain, worst_auroc_gain=worst_gain, clean_ap_gain=ap_gain,
                minimum_clean_and_worst_gain=.0005, requires_non_decreasing_clean_ap=True)
