"""Freeze a training-only internal validation split before fitting v5 models."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data import load_rows
from internal_split import make_manifest,partition_rows
from utils import save_json


def main():
    raw=ROOT/'data/raw/glaucoma_first'
    rows=load_rows(raw,'train');content=(raw/'train.csv').read_bytes()
    target=ROOT/'reports/v5_internal_split.json'
    if not target.exists():save_json(target,make_manifest(rows,content))
    training,holdout,fingerprint=partition_rows(rows,content,target)
    report=dict(train=len(training),holdout=len(holdout),
                train_positive=sum(int(r['True_Label']) for r in training),
                holdout_positive=sum(int(r['True_Label']) for r in holdout),manifest_sha256=fingerprint,
                source='official train only',patient_independence_verified=False)
    save_json(ROOT/'reports/v5_internal_split_summary.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
