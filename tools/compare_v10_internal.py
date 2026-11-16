"""Fixed actual-Age/missing-Age comparison on the original train holdout."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from clinical import AgeFundusDataset,build_age_model,age_gate
from data import FundusDataset,load_rows,audit_inputs
from internal_split import partition_rows
from model import build_model,probabilities_from_logits
from provenance import digest
from utils import metrics,save_json
from v8_integrity import control,verify_evidence
from v10_integrity import recipe,validate_completed_age,frozen,verify_reference_probabilities


def evaluate(saved,dataset,age_model):
    model=(build_age_model() if age_model else build_model(architecture='convnext_tiny')).cuda().eval().requires_grad_(False)
    model.load_state_dict(saved['model']);names=[];labels=[];probabilities=[]
    with torch.inference_mode():
        for step,batch in enumerate(DataLoader(dataset,batch_size=16,num_workers=4,shuffle=False,pin_memory=True)):
            if age_model:images,ages,y,batch_names=batch;ages=ages.cuda(non_blocking=True)
            else:images,y,batch_names=batch;ages=None
            images=images.cuda(non_blocking=True)
            with torch.autocast('cuda'):
                p=probabilities_from_logits(model(images,ages) if age_model else model(images).flatten())
                q=probabilities_from_logits(model(images.flip(-1),ages) if age_model else model(images.flip(-1)).flatten())
            probabilities.extend(((p+q)/2).cpu().tolist());labels.extend(y.tolist());names.extend(batch_names)
            if step%100==0:print('evaluated',len(names),'/',len(dataset),flush=True)
    del model;torch.cuda.empty_cache()
    return names,labels,probabilities


def main():
    output=ROOT/'output/v10';target=output/'internal_selection.json'
    if target.exists():raise FileExistsError('Preserve v10 comparison')
    record=frozen(ROOT);raw=ROOT/'data/raw/glaucoma_first';cache=ROOT/'data/cache384'
    audit=audit_inputs(raw,cache,384)
    _,held,split_hash=partition_rows(load_rows(raw,'train'),(raw/'train.csv').read_bytes(),ROOT/'reports/v5_internal_split.json')
    old=control(ROOT);new=validate_completed_age(ROOT/'runs/v10_inner_age',recipe('runs/v10_inner_age'),ROOT)
    evidence=list(record['evidence'])
    for name in ('best.pt','config.json','age_stats.json','best_metrics.json','status.json','split_binding.json'):
        path=ROOT/'runs/v10_inner_age'/name;evidence.append(dict(path=str(path),sha256=digest(path)))
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    baseline=FundusDataset(raw,'train',384,cache);baseline.rows=held
    datasets=[('reference',old['checkpoint'],baseline,False),
              ('actual_age',new['checkpoint'],AgeFundusDataset(raw,'train',384,cache,held,new['checkpoint']['age_stats']),True),
              ('missing_age',new['checkpoint'],AgeFundusDataset(raw,'train',384,cache,held,new['checkpoint']['age_stats'],force_missing=True),True)]
    with np.load(ROOT/'output/v8/internal_predictions.npz',allow_pickle=False) as archive:
        expected_names=archive['names'].tolist();expected_labels=archive['labels'].tolist()
        expected_reference=archive['v5_inner_convnext_clean_tta'].tolist()
    reference_record=json.loads((ROOT/'output/v8/internal_selection.json').read_text(encoding='utf-8'))['reference']['clean']
    values={};arrays={}
    for label,saved,dataset,is_age in datasets:
        names,labels,p=evaluate(saved,dataset,is_age)
        if names!=expected_names or labels!=expected_labels:raise ValueError('Internal Age comparison rows misaligned')
        if label=='reference':reference_max_delta=verify_reference_probabilities(p,expected_reference)
        values[label]=metrics(labels,p);arrays[label]=np.array(p)
        print(label,json.dumps(values[label]),flush=True)
    if abs(values['reference']['auroc']-reference_record['auroc'])>3e-5 or abs(values['reference']['average_precision']-reference_record['average_precision'])>3e-5:
        raise ValueError('Original control failed reproduction')
    gate=age_gate(values['reference'],values['actual_age'],values['missing_age'])
    path=output/'internal_predictions.npz'
    with path.open('xb') as stream:np.savez_compressed(stream,names=np.array(names),labels=np.array(labels),**arrays)
    evidence.append(dict(path=str(path),sha256=digest(path)))
    if audit_inputs(raw,cache,384)!=audit:raise ValueError('Age comparison image audit changed')
    verify_evidence(evidence)
    report=dict(metrics=values,gate=gate,selected_epoch=new['checkpoint']['metrics']['epoch'],
        age_stats=new['checkpoint']['age_stats'],manifest_sha256=split_hash,evidence=evidence,rows=len(labels),
        positive_rows=int(sum(labels)),reference_max_probability_delta=reference_max_delta,reference_probability_tolerance=1e-7,
        source='reused train holdout; no officialvalid/test metric use',
        limitation='Exploratory; patient grouping unavailable; Age/missingness may be cohort confounds')
    save_json(target,report);print('V10 INTERNAL GATE',json.dumps(gate),flush=True)


if __name__=='__main__':main()
