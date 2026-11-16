"""Pure score/CSV contracts and atomic output helpers."""
import csv
import json
import math
import os
from pathlib import Path

from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, auc


def metrics(labels, probabilities):
    if len(set(labels)) != 2 or len(labels) != len(probabilities):
        raise ValueError('Metrics require aligned observations with both classes')
    if not all(math.isfinite(float(p)) and 0 <= float(p) <= 1 for p in probabilities):
        raise ValueError('Invalid probabilities')
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    return {'auroc': float(roc_auc_score(labels, probabilities)),
            'average_precision': float(average_precision_score(labels, probabilities)),
            'auprc_trapezoid': float(auc(recall, precision))}


def is_better(candidate, previous):
    return previous is None or (candidate['auroc'], candidate['average_precision']) > (previous['auroc'], previous['average_precision'])


def write_submission(path, names, probabilities):
    if not names or len(names) != len(probabilities) or len(set(names)) != len(names):
        raise ValueError('Empty, duplicate or misaligned submission')
    if not all(math.isfinite(float(p)) and 0 <= float(p) <= 1 for p in probabilities):
        raise ValueError('Invalid probability')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image_name', 'Probability', 'Pred_Label'])
        writer.writerows((name, format(float(p), '.10g'), int(float(p) >= .5)) for name, p in zip(names, probabilities))
    os.replace(temporary, path)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


class Lifecycle:
    allowed = {'ready': {'training'}, 'training': {'validating', 'failed'},
               'validating': {'saving', 'failed'}, 'saving': {'training', 'complete', 'failed'},
               'complete': set(), 'failed': set()}

    def __init__(self):
        self.state = 'ready'

    def transition(self, next_state):
        if next_state not in self.allowed[self.state]:
            raise ValueError('ILLEGAL_TRANSITION: ' + self.state + ' -> ' + next_state)
        self.state = next_state
