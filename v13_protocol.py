"""Finite campaign transitions and resource/deadline guards."""
import copy
import math
import time
import json
from pathlib import Path
from provenance import digest
from v13_core import promotion

ALLOWED = {'implementation': {'preflight'}, 'preflight': {'fold_training'},
           'fold_training': {'comparison'}, 'comparison': {'full_training'},
           'full_training': {'finalization'}, 'finalization': {'awaiting_review'},
           'awaiting_review': {'complete'},
           'complete': set(), 'failed': set(), 'not_promoted': set(), 'budget_exceeded': set()}
TERMINAL = {'complete', 'failed', 'not_promoted', 'budget_exceeded'}


def transition(state, phase, now=None, **details):
    current = state['phase']
    if current not in ALLOWED or current in TERMINAL or (phase not in ALLOWED[current] and phase not in TERMINAL-{'complete'}):
        raise ValueError('ILLEGAL_TRANSITION: '+current+' -> '+phase)
    result = copy.deepcopy(state)
    result.update(details)
    result.update(phase=phase, revision=state['revision']+1, updated_at=time.time() if now is None else now)
    result['history'].append(dict(phase=phase, revision=result['revision'], time=result['updated_at'], details=details))
    return result


def remaining_timeout(deadline, now, per_process):
    remaining = min(deadline-now, per_process)
    if remaining <= 0: raise TimeoutError('Campaign budget exhausted')
    return remaining


def resource_gate(peak, milliseconds, delta, size):
    return (all(math.isfinite(v) for v in (peak, milliseconds, delta, size))
            and 0 < peak < 5_000_000_000 and 0 < milliseconds <= 100
            and 0 <= delta <= 1e-7 and 0 < size <= 500_000_000)


def stage_guard(stage, state, comparison):
    expected = {'preflight':'preflight','finalize':'finalization'}
    if stage not in expected or state['phase'] != expected[stage]:
        raise ValueError('Stage entry is not authorized by coordinator state')
    if stage == 'finalize' and (not comparison or comparison.get('passed') is not True
                                or comparison.get('fixed_epoch') != 8):
        raise ValueError('Finalization requires qualified complete three-fold comparison')


def require_promotion(root):
    root=Path(root)
    ticket=json.loads((root/'output/v13/promotion.json').read_text(encoding='utf-8'))
    for item in ticket['evidence']:
        if digest(item['path'])!=item['sha256']:raise ValueError('Promotion evidence changed')
    path=root/'output/v13/comparison.json'
    if digest(path)!=ticket['comparison_sha256']:raise ValueError('Comparison identity changed')
    report=json.loads(path.read_text(encoding='utf-8'))
    if (report.get('fixed_epoch')!=8 or report.get('rows')!=51396 or report.get('passed') is not True
            or not promotion(report['auroc_deltas'],report['oof_metrics']['control']['average_precision'],
                             report['oof_metrics']['mixup']['average_precision'])):
        raise ValueError('Promotion guards failed')
    return report


def training_guard(state, lock_owner, parent_pid, run_name, fold, treatment, full, now=None):
    now=time.time() if now is None else now
    expected_phase='full_training' if full else 'fold_training'
    expected_run='v13_full_mixup' if full else 'v13_fold'+str(fold)+'_'+treatment
    deadline=state.get('full_deadline' if full else 'deadline',0)
    if (state.get('phase')!=expected_phase or str(state.get('pid'))!=lock_owner.strip()
            or state.get('pid')!=parent_pid or run_name!=expected_run or now>=deadline):
        raise ValueError('Training admission rejected: phase, owner, run or deadline')
