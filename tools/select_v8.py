import json
from pathlib import Path
from select_optimized import main
from bootstrap_comparison import main as bootstrap
from v8_integrity import frozen,recipe,verify_evidence
from v7_integrity import validate_completed_run
from robustness import conservative_upgrade
from provenance import digest,inference_fingerprint
from utils import save_json

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    output=ROOT/'output/v8';frozen(ROOT)
    selection=json.loads((output/'internal_selection.json').read_text());verify_evidence(selection['evidence'])
    if selection['selected'] is None:raise ValueError('Internal robustness gate did not pass')
    validate_completed_run(ROOT/'runs/v8_full_color',recipe('v8_full_color',20261040,False,selection['selected']['epoch']),ROOT)
    main(output,ROOT/'output/v6/selected_config.json',True,'v6_fixed_equal_r50_convnext_horizontal_tta',local_limit_ms=100)
    bootstrap(output,labels_path=ROOT/'data/raw/glaucoma_first/valid.csv')
    uncertainty=json.loads((output/'bootstrap_comparison.json').read_text())
    passed=conservative_upgrade(uncertainty['auroc_gain'],uncertainty['auroc_gain_descriptive_95_interval'][0])
    config=output/'selected_config.json'
    if not passed:
        config=config.rename(output/'selected_config.validation_only.json')
    evidence=inference_fingerprint(config)
    for name in ['validation_predictions.csv','baseline_single_image_validation.csv','bootstrap_comparison.json','selection_report.json']:
        path=output/name;evidence.append(dict(path=str(path),sha256=digest(path)))
    label_path=ROOT/'data/raw/glaucoma_first/valid.csv';evidence.append(dict(path=str(label_path),sha256=digest(label_path)))
    save_json(output/'validation_ci_gate.json',dict(passed=passed,minimum_gain=.0005,requires_positive_descriptive_lower_bound=True,
              auroc_gain=uncertainty['auroc_gain'],descriptive_interval=uncertainty['auroc_gain_descriptive_95_interval'],evidence=evidence,
              limitation='Post-selection descriptive engineering filter, not independent or selection-adjusted significance'))
    if not passed:
        report=json.loads((output/'selection_report.json').read_text())
        report['rejection_reason']='V8 conservative filter failed: require AUROC gain >= .0005 and descriptive lower bound > 0; no test prediction'
        save_json(output/'selection_rejected.json',report)
        raise RuntimeError(report['rejection_reason'])
