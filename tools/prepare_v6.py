"""One predeclared equal-weight pair; validation only, no test-file access."""
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data import load_rows
from utils import metrics,save_json


def main():
    output=ROOT/'output/v6'
    if output.exists():raise FileExistsError('Preserve existing v6 output')
    reference=ROOT/'output/v5/baseline_single_image_validation.csv'
    candidate=ROOT/'output/v5/validation_predictions.csv'
    checkpoints=[ROOT/'runs/v3_r50_384_w1/best.pt',ROOT/'runs/v5_full/best.pt']
    evidence=[reference,candidate,ROOT/'data/raw/glaucoma_first/valid.csv',ROOT/'output/v3/selected_config.json',
              ROOT/'output/v5/candidate_00.json',ROOT/'output/v5/selection_rejected.json',*checkpoints]
    hashes=[dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in evidence]
    output.mkdir()
    save_json(output/'input_fingerprints.json',hashes)
    labels=load_rows(ROOT/'data/raw/glaucoma_first','valid')
    with reference.open(encoding='utf-8',newline='') as stream:a=list(csv.DictReader(stream))
    with candidate.open(encoding='utf-8',newline='') as stream:b=list(csv.DictReader(stream))
    names=[row['Image_name'] for row in labels]
    if len(names)!=5849 or [r['Image_name'] for r in a]!=names or [r['Image_name'] for r in b]!=names:
        raise ValueError('Validation names/order mismatch')
    for version,checkpoint,configuration in [('v3',checkpoints[0],ROOT/'output/v3/selected_config.json'),
                                              ('v5',checkpoints[1],ROOT/'output/v5/candidate_00.json')]:
        spec=json.loads(configuration.read_text())
        if len(spec['models'])!=1 or spec['models'][0]['tta'] is not True or (configuration.parent/spec['models'][0]['checkpoint']).resolve()!=checkpoint:
            raise ValueError('Predeclared member mismatch: '+version)
    probabilities=[(float(x['Probability'])+float(y['Probability']))/2 for x,y in zip(a,b)]
    metric=metrics([int(row['True_Label']) for row in labels],probabilities)
    name='v6_fixed_equal_v3_r50_and_v5_convnext_horizontal_tta'
    item=dict(name=name,models=[dict(checkpoint=str(path),tta=True) for path in checkpoints],
              **metric,selection_source='one predeclared equal-weight pair; cached raw-path CSV probabilities, subject to raw-image recheck')
    spec=dict(schema_version=1,name=name,models=[dict(checkpoint='../../runs/'+path.parent.name+'/best.pt',tta=True) for path in checkpoints],
              validation_metrics=metric,batch=1,precision='fp16',probability_precision='fp32')
    save_json(output/'candidate_00.json',spec)
    save_json(output/'candidate_comparison.json',[item])
    save_json(ROOT/'reports/v6_status.json',dict(phase='candidates_compared'))
    print(json.dumps(item),flush=True)


if __name__=='__main__':main()
