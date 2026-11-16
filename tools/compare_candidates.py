"""Validation-only model/TTA/ensemble selection; no test data are opened."""
import csv
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data import FundusDataset
from model import build_model,probabilities_from_logits
from utils import metrics,save_json
from cache_paths import cache_path

RUNS=['baseline_r18_224','opt_r18_224_w2','opt_r18_384_w2','opt_r18_384_w1']


def combine_tta(plain,flipped,tta):
    return ((np.array(plain,dtype=np.float32)+np.array(flipped,dtype=np.float32))/2).astype(np.float64) if tta else np.array(plain)


def main(output=None):
    torch.set_num_threads(4)
    device='cuda'
    candidates={}
    reference_names=None
    reference_labels=None
    details={}
    for name in RUNS:
        checkpoint=ROOT/'runs'/name/'best.pt'
        state=json.loads((checkpoint.parent/'status.json').read_text(encoding='utf-8'))
        if state['state']!='complete':
            raise RuntimeError('Candidate training incomplete: '+name)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        size=saved['size']
        layout=torch.channels_last if saved.get('memory_format','contiguous')=='channels_last' else torch.contiguous_format
        model=build_model(architecture=saved.get('architecture','resnet18')).to('cuda',memory_format=layout).eval()
        model.load_state_dict(saved['model'])
        dataset=FundusDataset(ROOT/'data/raw/glaucoma_first','valid',size,cache_path(size))
        loader=DataLoader(dataset,batch_size=1,num_workers=4,shuffle=False,pin_memory=True)
        plain,flipped,labels,names=[],[],[],[]
        with torch.inference_mode():
            for images,y,batch_names in loader:
                images=images.to(device,non_blocking=True,memory_format=layout)
                with torch.autocast(device_type='cuda'):
                    a=probabilities_from_logits(model(images).flatten()).cpu().numpy()
                    b=probabilities_from_logits(model(images.flip(-1)).flatten()).cpu().numpy()
                plain.extend(a.tolist());flipped.extend(b.tolist());labels.extend(y.tolist());names.extend(batch_names)
        if reference_names is None:
            reference_names,reference_labels=names,labels
        if names!=reference_names or labels!=reference_labels:
            raise ValueError('Candidate validation alignment mismatch')
        for tta in [False,True]:
            key=name+('_tta' if tta else '')
            # Match Predictor's FP32 TTA average, then Python-float member fusion.
            probabilities=combine_tta(plain,flipped,tta)
            candidates[key]=probabilities
            details[key]=[{'checkpoint':str(checkpoint),'tta':tta}]
        del model
        torch.cuda.empty_cache()
        print('Validation evaluated',name,flush=True)
    singles=list(candidates)
    # Predeclared equal-weight pairs, matching TTA mode, different training runs.
    for a,b in itertools.combinations(singles,2):
        if a.endswith('_tta')!=b.endswith('_tta') or details[a][0]['checkpoint']==details[b][0]['checkpoint']:
            continue
        key=a+' + '+b
        candidates[key]=(candidates[a]+candidates[b])/2
        details[key]=details[a]+details[b]
    results=[dict(name=name,**metrics(reference_labels,p),models=details[name]) for name,p in candidates.items()]
    results.sort(key=lambda x:(x['auroc'],x['average_precision']),reverse=True)
    output=Path(output) if output is not None else ROOT/'output'/'optimized'
    output.mkdir(parents=True,exist_ok=True)
    save_json(output/'candidate_comparison.json',results)
    for rank,item in enumerate(results):
        spec={'schema_version':1,'models':[{'checkpoint':os.path.relpath(member['checkpoint'],output),
                                          'tta':member['tta']} for member in item['models']],
              'validation_metrics':{k:item[k] for k in ('auroc','average_precision','auprc_trapezoid')},
              'name':item['name'],'batch':1,'precision':'fp16','probability_precision':'fp32'}
        save_json(output/f'candidate_{rank:02}.json',spec)
    with (output/'validation_candidates.npz').open('wb') as file:
        np.savez_compressed(file,labels=np.array(reference_labels),names=np.array(reference_names),
                            **{f'candidate_{rank:02}':candidates[item['name']] for rank,item in enumerate(results)})
    print(json.dumps(results[:5]),flush=True)


if __name__=='__main__':
    main()
