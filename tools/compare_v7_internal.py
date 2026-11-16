"""Internal-only comparison of speed-qualified EfficientNet/ConvNeXt recipes."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data import FundusDataset,load_rows
from internal_split import partition_rows
from model import build_model,probabilities_from_logits
from utils import save_json
from v7_protocol import rank_candidates
from provenance import inference_fingerprint
from v7_integrity import requested_recipe,validate_completed_run


def main():
    output=ROOT/'output/v7';output.mkdir(exist_ok=True)
    raw=ROOT/'data/raw/glaucoma_first';split=ROOT/'reports/v5_internal_split.json'
    _,held,fingerprint=partition_rows(load_rows(raw,'train'),(raw/'train.csv').read_bytes(),split)
    dataset=FundusDataset(raw,'train',384,ROOT/'data/cache384');dataset.rows=held
    torch.set_num_threads(4);results={};evidence=[];reference_names=None;reference_labels=None
    capacity=json.loads((ROOT/'reports/v7_capacity.json').read_text())
    latency=json.loads((ROOT/'reports/v7_latency_probe.json').read_text())
    eligible=latency['eligible']
    fixed_sources=[split,ROOT/'reports/v7_capacity.json',ROOT/'reports/v7_latency_probe.json',
                   ROOT/'reports/pretrain_provenance_v7.json',ROOT/'runs/v5_full/config.json',ROOT/'runs/v5_full/best.pt']
    fixed_sources+=[Path(item['path']) for item in inference_fingerprint(ROOT/'output/v6/selected_config.json')]
    for source in dict.fromkeys(fixed_sources):
        evidence.append(dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    frozen_recipe=None
    for run in ['v7_inner_efficientnet','v5_inner_convnext']:
        directory=ROOT/'runs'/run
        if json.loads((directory/'status.json').read_text())['state']!='complete':raise ValueError('Incomplete internal training')
        config=json.loads((directory/'config.json').read_text())
        saved=torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
        if run=='v7_inner_efficientnet':
            checked=validate_completed_run(directory,requested_recipe('runs/'+run,20261033,capacity['selected_batch'],True),ROOT)
            frozen_recipe=checked['config']
        if config['init_checkpoint'] is not None or saved['split_manifest_sha256']!=fingerprint or saved['validation_partition']!='internal_train_holdout':
            raise ValueError('Exposed or incompatible internal checkpoint')
        for source in [directory/'best.pt',directory/'config.json',directory/'split_binding.json']:
            evidence.append(dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        model=build_model(architecture=config['architecture']).cuda().eval();model.load_state_dict(saved['model'])
        plain=[];flipped=[];labels=[];names=[]
        with torch.inference_mode():
            for images,y,batch_names in DataLoader(dataset,batch_size=1,num_workers=4,shuffle=False,pin_memory=True):
                images=images.cuda(non_blocking=True)
                with torch.autocast('cuda'):
                    a=probabilities_from_logits(model(images).flatten());b=probabilities_from_logits(model(images.flip(-1)).flatten())
                plain.extend(a.cpu().tolist());flipped.extend(b.cpu().tolist());labels.extend(y.tolist());names.extend(batch_names)
        if reference_names is None:reference_names,reference_labels=names,labels
        elif names!=reference_names or labels!=reference_labels:raise ValueError('Internal prediction alignment mismatch')
        results[run]=dict(plain=plain,flipped=flipped,epoch=saved['metrics']['epoch'])
        del model;torch.cuda.empty_cache();print('Internal validation evaluated',run,flush=True)
    for item in evidence:
        if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Frozen internal/reference evidence changed during comparison')
    new=results['v7_inner_efficientnet'];anchor=results['v5_inner_convnext']
    report=rank_candidates(reference_labels,new['plain'],new['flipped'],anchor['plain'],anchor['flipped'],eligible)
    if report['selected'] is not None:report['selected']['efficientnet_epoch']=new['epoch']
    previous=json.loads((ROOT/'output/v5/internal_selection.json').read_text())['selected']
    if abs(report['reference']['auroc']-previous['auroc'])>1e-12:raise ValueError('Fixed internal ConvNeXt reference did not reproduce')
    report.update(schema_version=1,manifest_sha256=fingerprint,evidence=evidence,frozen_recipe=frozen_recipe,
                  source='reused official train internal holdout only; not a new independent test',patient_independence_verified=False)
    save_json(output/'internal_selection.json',report)
    with (output/'internal_predictions.npz').open('wb') as stream:
        np.savez_compressed(stream,names=np.array(reference_names),labels=np.array(reference_labels),
                            new_plain=new['plain'],new_flipped=new['flipped'],anchor_plain=anchor['plain'],anchor_flipped=anchor['flipped'])
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
