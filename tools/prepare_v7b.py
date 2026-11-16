"""Freeze exactly the documented internally ranked plain-pair resource fallback."""
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from provenance import inference_fingerprint
from utils import save_json
from v7_protocol import resource_fallback
from v7_integrity import requested_recipe,validate_completed_run


def main():
    original=ROOT/'output/v7';output=ROOT/'output/v7b'
    selection=json.loads((original/'internal_selection.json').read_text())
    chosen=resource_fallback(selection)
    for item in selection['evidence']:
        if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Original internal evidence changed')
    validate_completed_run(ROOT/'runs/v7_full_efficientnet',
                           requested_recipe('runs/v7_full_efficientnet',20261034,selection['frozen_recipe']['batch'],False,
                                            chosen['efficientnet_epoch']),ROOT)
    timing=json.loads((original/'local_timing_candidates.json').read_text())
    fusion=json.loads((ROOT/'reports/v7_fusion_probe.json').read_text())
    if len(timing)!=1 or timing[0]['mean_ms']<=100 or fusion['within_limit']:
        raise ValueError('Expected preserved failed original and fusion resource trials')
    output.mkdir(exist_ok=False)
    (output/'internal_selection.json').write_bytes((original/'internal_selection.json').read_bytes())
    members=[dict(checkpoint='../../runs/v7_full_efficientnet/best.pt',tta=False),
             dict(checkpoint='../../runs/v5_full/best.pt',tta=False)]
    name='v7b_fixed_internal_second_pair_plain'
    save_json(output/'candidate_00.json',dict(schema_version=1,name=name,models=members,batch=1,
              precision='fp16',probability_precision='fp32',selection_source='OPTIMIZATION-v7b: fixed internal second-rank resource fallback'))
    save_json(output/'candidate_comparison.json',[dict(name=name,models=[dict(checkpoint=str((output/item['checkpoint']).resolve()),tta=False) for item in members],
              internal_validation_metrics={key:chosen[key] for key in ('auroc','average_precision','auprc_trapezoid')},
              selection_source='Frozen internal ranking only; no official-valid or leaderboard selection')])
    sources=[original/'internal_selection.json',original/'candidate_00.json',original/'local_timing_candidates.json',
             ROOT/'reports/v7_selection.log',ROOT/'reports/v7_finalization_status.json',
             ROOT/'reports/v7_latency_diagnosis.json',ROOT/'reports/v7_fusion_probe.json',
             ROOT/'docs/OPTIMIZATION-v7b.md',ROOT/'predict.py',ROOT/'model.py',
             output/'internal_selection.json',output/'candidate_comparison.json']
    sources += [Path(item['path']) for item in inference_fingerprint(output/'candidate_00.json')]
    evidence=[dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in dict.fromkeys(sources)]
    save_json(output/'deployment_revision.json',dict(schema_version=1,kind='fixed_internal_second_pair_plain',
              previous_failure_ms=timing[0]['mean_ms'],rejected_fusion_ms=fusion['mean_ms'],
              chosen=chosen,evidence=evidence,official_T4_verified=False))
    save_json(ROOT/'reports/v7b_status.json',dict(phase='candidates_compared',source='completed v7 weights; fixed resource fallback; no retraining'))
    print(json.dumps(chosen),flush=True)


if __name__=='__main__':main()
