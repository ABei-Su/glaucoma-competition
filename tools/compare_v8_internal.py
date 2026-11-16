"""Clean and two predeclared color stresses, train holdout only."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data import FundusDataset,load_rows,audit_inputs
from internal_split import partition_rows
from model import build_model,probabilities_from_logits
from provenance import digest
from robustness import CONDITIONS,StressTransform,choose_robust_candidate
from utils import metrics,save_json
from v8_integrity import frozen,control,recipe,verify_evidence
from v7_integrity import validate_completed_run


def main():
    record=frozen(ROOT);output=ROOT/'output/v8'
    if (output/'internal_selection.json').exists():raise FileExistsError('Preserve completed comparison')
    control(ROOT)
    audit_inputs(ROOT/'data/raw/glaucoma_first',ROOT/'data/cache384',384)
    checked=validate_completed_run(ROOT/'runs/v8_inner_color',recipe('v8_inner_color',20261031,True),ROOT)
    raw=ROOT/'data/raw/glaucoma_first';split=ROOT/'reports/v5_internal_split.json'
    _,held,split_hash=partition_rows(load_rows(raw,'train'),(raw/'train.csv').read_bytes(),split)
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    evidence=list(record['evidence']);all_results={};arrays={};expected_names=None;expected_labels=None
    for run in ['v5_inner_convnext','v8_inner_color']:
        checkpoint=ROOT/'runs'/run/'best.pt'
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if saved['split_manifest_sha256']!=split_hash or saved['validation_partition']!='internal_train_holdout':
            raise ValueError('Incompatible/exposed internal checkpoint')
        for filename in ['best.pt','config.json','best_metrics.json','split_binding.json','status.json']:
            path=checkpoint.parent/filename;evidence.append(dict(path=str(path),sha256=digest(path)))
        model=build_model(architecture='convnext_tiny').cuda().eval().requires_grad_(False);model.load_state_dict(saved['model'])
        result={}
        for condition in CONDITIONS:
            dataset=FundusDataset(raw,'train',384,ROOT/'data/cache384');dataset.rows=held;dataset.transform=StressTransform(condition)
            plain=[];tta=[];labels=[];names=[]
            with torch.inference_mode():
                for images,y,batch_names in DataLoader(dataset,batch_size=16,num_workers=4,shuffle=False,pin_memory=True):
                    images=images.cuda(non_blocking=True)
                    with torch.autocast('cuda'):
                        p=probabilities_from_logits(model(images).flatten());q=probabilities_from_logits(model(images.flip(-1)).flatten())
                        augmented=(p+q)/2
                    plain.extend(p.cpu().tolist());tta.extend(augmented.cpu().tolist());labels.extend(y.tolist());names.extend(batch_names)
            if expected_names is None:expected_names,expected_labels=names,labels
            elif names!=expected_names or labels!=expected_labels:raise ValueError('Unaligned stress predictions')
            result[condition]=dict(plain=metrics(labels,plain),tta=metrics(labels,tta))
            arrays[run+'_'+condition+'_plain']=np.array(plain);arrays[run+'_'+condition+'_tta']=np.array(tta)
            print(run,condition,json.dumps(result[condition]),flush=True)
        all_results[run]=result;del model;torch.cuda.empty_cache()
    reference={condition:all_results['v5_inner_convnext'][condition]['tta'] for condition in CONDITIONS}
    prior=json.loads((ROOT/'output/v5/internal_selection.json').read_text())['selected']['auroc']
    if abs(reference['clean']['auroc']-prior)>3e-5:raise ValueError('Fixed control failed clean reproducibility check')
    candidates={mode:{condition:all_results['v8_inner_color'][condition][mode] for condition in CONDITIONS} for mode in ['plain','tta']}
    report=choose_robust_candidate(reference,candidates,record['eligible'],checked['checkpoint']['metrics']['epoch'])
    path=output/'internal_predictions.npz'
    with path.open('wb') as stream:np.savez_compressed(stream,names=np.array(expected_names),labels=np.array(expected_labels),**arrays)
    evidence.append(dict(path=str(path),sha256=digest(path)));verify_evidence(evidence)
    report.update(evidence=evidence,manifest_sha256=split_hash,all_results=all_results,validation_batch=16,
                  patient_independence_verified=False,frozen_recipe=checked['config'])
    save_json(output/'internal_selection.json',report)
    print('INTERNAL SELECTED',json.dumps(report['selected']),flush=True)


if __name__=='__main__':main()
