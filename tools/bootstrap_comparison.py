"""Paired stratified validation bootstrap; descriptive after model selection."""
import csv
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils import save_json
from provenance import digest


def read(path):
    with path.open(encoding='utf-8',newline='') as stream:
        return list(csv.DictReader(stream))


def main(output=None,labels_path=None):
    output=Path(output) if output is not None else ROOT/'output/optimized'
    labels_path=Path(labels_path) if labels_path is not None else ROOT/'runs/baseline_r18_224/validation_predictions.csv'
    labels_sha256=digest(labels_path);labels=read(labels_path)
    baseline=read(output/'baseline_single_image_validation.csv')
    selected=read(output/'validation_predictions.csv')
    if [r['Image_name'] for r in baseline]!=[r['Image_name'] for r in selected]:
        raise ValueError('Unpaired validation predictions')
    if [r['Image_name'] for r in labels]!=[r['Image_name'] for r in baseline]:
        raise ValueError('Unpaired baseline labels')
    y=np.array([int(float(r['True_Label'])) for r in labels])
    a=np.array([float(r['Probability']) for r in baseline])
    b=np.array([float(r['Probability']) for r in selected])
    positive=np.flatnonzero(y==1);negative=np.flatnonzero(y==0)
    rng=np.random.default_rng(20261024)
    auc=[];ap=[]
    for _ in range(1000):
        indices=np.concatenate([rng.choice(positive,len(positive),replace=True),rng.choice(negative,len(negative),replace=True)])
        auc.append(roc_auc_score(y[indices],b[indices])-roc_auc_score(y[indices],a[indices]))
        ap.append(average_precision_score(y[indices],b[indices])-average_precision_score(y[indices],a[indices]))
    if digest(labels_path)!=labels_sha256:raise ValueError('Ground-truth labels changed during bootstrap')
    report=dict(replicates=1000,seed=20261024,positive=len(positive),negative=len(negative),labels_path=str(labels_path.resolve()),labels_sha256=labels_sha256,
                reference_name=json.loads((output/'selection_report.json').read_text(encoding='utf-8')).get('reference_name','baseline_r18_224'),
                auroc_gain=float(roc_auc_score(y,b)-roc_auc_score(y,a)),
                auroc_gain_descriptive_95_interval=np.quantile(auc,[.025,.975]).tolist(),
                ap_gain=float(average_precision_score(y,b)-average_precision_score(y,a)),
                ap_gain_descriptive_95_interval=np.quantile(ap,[.025,.975]).tolist(),
                probability_correlation=float(np.corrcoef(a,b)[0,1]),
                limitations='Post-selection image-level bootstrap; does not correct selection bias or patient clustering. Both models use final batch1 raw-image inference. Not clinical validation or official test performance.')
    save_json(output/'bootstrap_comparison.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
