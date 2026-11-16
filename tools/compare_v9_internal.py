"""One fixed .5/.5 weight soup evaluated on the existing train-only holdout."""
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import FundusDataset, load_rows, audit_inputs
from internal_split import partition_rows
from model import build_model, probabilities_from_logits
from provenance import digest, inference_fingerprint
from robustness import StressTransform
from train import atomic_save
from utils import metrics, save_json
from v7_integrity import validate_completed_run
from v8_integrity import control, frozen, recipe, verify_evidence
from weight_soup import CONDITIONS, average_pair, validate_pair, soup_gate
from v9_protocol import advance, require_fresh_experiment


def evaluate(saved, held, raw, condition):
    model = build_model(architecture='convnext_tiny').cuda().eval().requires_grad_(False)
    model.load_state_dict(saved['model'])
    if any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in model.modules()):
        raise ValueError('Unexpected BatchNorm in the fixed architecture')
    dataset = FundusDataset(raw, 'train', 384, ROOT/'data/cache384')
    dataset.rows = held
    dataset.transform = StressTransform(condition)
    probabilities, labels, names = [], [], []
    with torch.inference_mode():
        for index, (images, y, batch_names) in enumerate(DataLoader(dataset, batch_size=16, num_workers=4,
                                                                 shuffle=False, pin_memory=True)):
            images = images.cuda(non_blocking=True)
            with torch.autocast('cuda'):
                p = probabilities_from_logits(model(images).flatten())
                q = probabilities_from_logits(model(images.flip(-1)).flatten())
                p = (p + q) / 2
            probabilities.extend(p.cpu().tolist())
            labels.extend(y.tolist())
            names.extend(batch_names)
            if index % 100 == 0:
                print(condition, 'evaluated', len(names), '/', len(held), flush=True)
    del model
    torch.cuda.empty_cache()
    return names, labels, probabilities


def main():
    output = ROOT/'output/v9'
    output.mkdir(exist_ok=True)
    status = ROOT/'reports/v9_status.json'
    require_fresh_experiment(output, status)
    lock = output/'experiment.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    current = dict(phase='ready', revision=0, history=[])
    def update(phase, **details):
        nonlocal current
        current = advance(current, phase, str(current['revision'] + 1), details)
        save_json(status, current)
    try:
        update('verifying')
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        frozen(ROOT)
        previous = json.loads((ROOT/'output/v8/internal_selection.json').read_text(encoding='utf-8'))
        verify_evidence(previous['evidence'])
        left = control(ROOT)
        right = validate_completed_run(ROOT/'runs/v8_inner_color', recipe('v8_inner_color', 20261031, True), ROOT)
        validate_pair(left['config'], right['config'], left['checkpoint'], right['checkpoint'])
        raw = ROOT/'data/raw/glaucoma_first'
        audit = audit_inputs(raw, ROOT/'data/cache384', 384)
        _, held, split_hash = partition_rows(load_rows(raw, 'train'), (raw/'train.csv').read_bytes(),
                                             ROOT/'reports/v5_internal_split.json')
        if split_hash != previous['manifest_sha256'] or len(held) != 7710:
            raise ValueError('Internal split changed')
        # A matched-seed reconstruction records the common public backbone/head recipe;
        # historical initial tensors were not saved, so it is not byte-level proof of them.
        torch.manual_seed(left['config']['seed'])
        initial = build_model(ROOT/left['config']['pretrain'], architecture='convnext_tiny')
        head = hashlib.sha256(b''.join(tensor.detach().numpy().tobytes()
                                     for tensor in initial.classifier[-1].state_dict().values())).hexdigest()
        del initial
        sources = [ROOT/name for name in ('docs/OPTIMIZATION-v9.md', 'docs/ADR-008-fixed-weight-soup.md',
            'weight_soup.py', 'v9_protocol.py', 'tools/compare_v9_internal.py', 'tests/test_weight_soup.py', 'tests/test_v9_protocol.py',
            'docs/REVIEW-v9-preflight.md',
            'model.py', 'data.py', 'train.py', 'utils.py', 'robustness.py', 'v7_integrity.py', 'v8_integrity.py',
            'internal_split.py', 'provenance.py', 'requirements-local.lock.txt', 'reports/v9_verification.log',
            'reports/v5_internal_split.json', 'data/cache384/READY.json', 'data/raw/glaucoma_first/train.csv',
            'output/v8/internal_selection.json', 'output/v8/internal_predictions.npz', 'output/v8/frozen_inputs.json',
            left['config']['pretrain'])]
        for run in ('v5_inner_convnext', 'v8_inner_color', 'v5_full'):
            sources.extend(ROOT/'runs'/run/name for name in ('config.json', 'best.pt', 'best_metrics.json', 'status.json'))
        sources.append(Path(audit['manifest_path']))
        evidence = [dict(path=str(path), sha256=digest(path)) for path in sources]
        evidence.extend(inference_fingerprint(ROOT/'output/v6/selected_config.json'))
        record = dict(evidence=evidence, coefficients=[.5, .5], horizontal_tta=True, source_epochs=[8, 8],
                      initialization_seed=20261031, reconstructed_binary_head_sha256=head,
                      initialization_limitation='Matching recorded seed/public hash/current construction; historical initial head bytes unavailable',
                      environment=dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                                       device=torch.cuda.get_device_name(0)),
                      internal_manifest_sha256=split_hash, source='official train internal holdout only',
                      official_validation_evaluated=False, integrity_audit_includes_official_valid=True,
                      test_used=False, platform_score_used=False)
        target = output/'frozen_inputs.json'
        save_json(target, record)
        update('composing')
        derived = {key: left['checkpoint'][key] for key in ('architecture', 'size', 'memory_format',
            'validation_probability_precision', 'data_fingerprint', 'split_manifest_sha256', 'validation_partition')}
        derived.update(schema_version=1, threshold=.5, precision='fp16',
                       model=average_pair(left['checkpoint']['model'], right['checkpoint']['model']),
                       composition=dict(kind='fixed_equal_weight_soup', coefficients=[.5, .5], source_epochs=[8, 8],
                                        sources=[dict(path=str(ROOT/'runs'/run/'best.pt'),
                                                      sha256=digest(ROOT/'runs'/run/'best.pt'))
                                                 for run in ('v5_inner_convnext', 'v8_inner_color')],
                                        public_initialization_sha256=digest(ROOT/left['config']['pretrain']),
                                        initialization_seed=20261031))
        checkpoint_path = output/'internal_soup.pt'
        if checkpoint_path.exists():
            raise FileExistsError('Preserve existing derived soup; review interrupted experiment')
        atomic_save(derived, checkpoint_path)
        derived = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        evidence = evidence + [dict(path=str(checkpoint_path), sha256=digest(checkpoint_path))]
        archive = np.load(ROOT/'output/v8/internal_predictions.npz', allow_pickle=False)
        expected_names, expected_labels = archive['names'].tolist(), archive['labels'].tolist()
        reference = {condition: metrics(expected_labels, archive['v5_inner_convnext_'+condition+'_tta'].tolist())
                     for condition in CONDITIONS}
        if reference != previous['reference']:
            raise ValueError('Stored control predictions/metrics mismatch')
        update('evaluating', step='control_clean_reproduction')
        names, labels, probabilities = evaluate(left['checkpoint'], held, raw, 'clean')
        if names != expected_names or labels != expected_labels:
            raise ValueError('Control names/labels misaligned')
        reproduced = metrics(labels, probabilities)
        if abs(reproduced['auroc'] - reference['clean']['auroc']) > 3e-5 or abs(reproduced['average_precision'] - reference['clean']['average_precision']) > 3e-5:
            raise ValueError('Control failed clean AUROC/AP reproduction')
        values, arrays = {}, dict(names=np.array(names), labels=np.array(labels), control_clean=np.array(probabilities))
        for condition in CONDITIONS:
            update('evaluating', step='soup_'+condition)
            names, labels, probabilities = evaluate(derived, held, raw, condition)
            if names != expected_names or labels != expected_labels:
                raise ValueError('Soup names/labels misaligned')
            values[condition] = metrics(labels, probabilities)
            arrays['soup_'+condition] = np.array(probabilities)
            print('SOUP', condition, json.dumps(values[condition]), flush=True)
        path = output/'internal_predictions.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        evidence.append(dict(path=str(path), sha256=digest(path)))
        if audit_inputs(raw, ROOT/'data/cache384', 384) != audit:
            raise ValueError('Input content audit changed during v9 evaluation')
        verify_evidence(evidence)
        report = dict(reference=reference, candidate=values, control_reproduced=reproduced,
                      gate=soup_gate(reference, values), evidence=evidence, coefficients=[.5, .5],
                      horizontal_tta=True, rows=len(labels), positives=int(sum(labels)),
                      source='reused official-train internal holdout; synthetic stresses; exploratory',
                      official_validation_evaluated=False, integrity_audit_includes_official_valid=True,
                      test_used=False, platform_score_used=False, image_content_audited_before_and_after=True)
        save_json(output/'internal_selection.json', report)
        phase = 'internal_passed' if report['gate']['passed'] else 'not_promoted'
        update(phase, gate=report['gate'],
               reason='Fixed 50:50 clean/worst AUROC and clean AP gate', test_csv_created=False)
        print('V9 INTERNAL GATE', json.dumps(report['gate']), flush=True)
    except BaseException as error:
        update('failed', error=str(error))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
