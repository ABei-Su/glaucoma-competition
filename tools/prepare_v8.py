import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from provenance import digest,inference_fingerprint
from v8_integrity import control,verify_evidence
from utils import save_json


def main():
    output=ROOT/'output/v8';target=output/'frozen_inputs.json'
    if target.exists():raise FileExistsError('Preserve frozen v8 inputs')
    speed=json.loads((ROOT/'reports/v8_latency_probe.json').read_text());verify_evidence(speed['evidence'])
    if not any(speed['eligible'].values()):raise ValueError('No resource-qualified v8 shape')
    control(ROOT)
    sources=[ROOT/name for name in ['data.py','train.py','model.py','robustness.py','v8_integrity.py','v7_integrity.py',
        'docs/OPTIMIZATION-v8.md','docs/ADR-007-training-augmentation-profile.md',
        'tools/run_v8.py','tools/compare_v8_internal.py','reports/v8_latency_probe.json','reports/v5_capacity.json',
        'reports/v5_internal_split.json','reports/pretrain_provenance_v5.json','data/cache384/READY.json',
        'output/v5/internal_selection.json','data/raw/glaucoma_first/valid.csv',
        'pretrain_model/convnext_tiny-983f1562.pth','runs/v3_r50_384_w2/best.pt']]
    for name in ['config.json','best.pt','best_metrics.json','split_binding.json','status.json']:
        sources.append(ROOT/'runs/v5_inner_convnext'/name)
    sources += [Path(item['path']) for item in inference_fingerprint(ROOT/'output/v6/selected_config.json')]
    record=dict(schema_version=1,eligible=speed['eligible'],batch=16,matched_control='v5_inner_convnext',
                evidence=[dict(path=str(path),sha256=digest(path)) for path in dict.fromkeys(sources)],
                source='official train internal holdout only; public ImageNet; resource probe has no label use')
    save_json(target,record);print('V8 INPUTS FROZEN',flush=True)


if __name__=='__main__':main()
