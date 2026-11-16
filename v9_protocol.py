"""Pure, bounded state transitions for the single-process v9 comparison."""
import copy
from pathlib import Path

ALLOWED = {
    'ready': {'verifying'},
    'verifying': {'composing', 'failed'},
    'composing': {'evaluating', 'failed'},
    'evaluating': {'evaluating', 'internal_passed', 'not_promoted', 'failed'},
    'internal_passed': set(), 'not_promoted': set(), 'failed': set(),
}


def require_fresh_experiment(output, status):
    output, status = Path(output), Path(status)
    if status.exists() or (output.exists() and any(output.iterdir())):
        raise FileExistsError('Preserve prior v9 status/evidence; inspect PID and archive manually before restart')


def advance(state, phase, event_id, details):
    if not event_id or set(details) & {'phase', 'revision', 'history'}:
        raise ValueError('Invalid experiment event')
    event = dict(event_id=event_id, phase=phase, details=details)
    history = state['history']
    existing = [item for item in history if item['event_id'] == event_id]
    if existing:
        if existing == [event] and history[-1] == event:
            return copy.deepcopy(state)
        raise ValueError('Duplicate or out-of-order experiment event')
    if phase not in ALLOWED.get(state['phase'], set()):
        raise ValueError('ILLEGAL_V9_TRANSITION: ' + state['phase'] + ' -> ' + phase)
    return dict(phase=phase, revision=state['revision'] + 1,
                history=copy.deepcopy(history + [event]), **details)
