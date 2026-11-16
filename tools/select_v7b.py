import hashlib
import json
from pathlib import Path
from select_optimized import main
from v7_integrity import requested_recipe,validate_completed_run

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    output=ROOT/'output/v7b'
    selection=json.loads((output/'internal_selection.json').read_text())
    revision=json.loads((output/'deployment_revision.json').read_text())
    for item in selection['evidence']+revision['evidence']:
        if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Frozen internal/deployment evidence changed')
    validate_completed_run(ROOT/'runs/v7_full_efficientnet',
                           requested_recipe('runs/v7_full_efficientnet',20261034,selection['frozen_recipe']['batch'],False,
                                            selection['selected']['efficientnet_epoch']),ROOT)
    main(output,ROOT/'output/v6/selected_config.json',True,'v6_fixed_equal_r50_convnext_horizontal_tta',local_limit_ms=100)
