"""Select exactly four predeclared candidates using only the internal TRAIN holdout."""
import hashlib
import json
import os
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
from utils import metrics,save_json


def main():
    output=ROOT/'output/v5';output.mkdir(exist_ok=True)
    raw=ROOT/'data/raw/glaucoma_first'
    _,held,fingerprint=partition_rows(load_rows(raw,'train'),(raw/'train.csv').read_bytes(),ROOT/'reports/v5_internal_split.json')
    dataset=FundusDataset(raw,'train',384,ROOT/'data/cache384')
    dataset.rows=held
    torch.set_num_threads(4)
    candidates=[];reference_names=None;evidence=[]
    for run in ['v5_inner_r50','v5_inner_convnext']:
        directory=ROOT/'runs'/run
        if json.loads((directory/'status.json').read_text())['state']!='complete':raise ValueError('Incomplete run')
        config=json.loads((directory/'config.json').read_text())
        for source in [directory/'config.json',directory/'best.pt',directory/'split_binding.json']:
            evidence.append(dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        saved=torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
        if config['init_checkpoint'] is not None or saved['split_manifest_sha256']!=fingerprint or saved['validation_partition']!='internal_train_holdout':
            raise ValueError('Internal comparison provenance mismatch')
        model=build_model(architecture=config['architecture']).cuda().eval()
        model.load_state_dict(saved['model'])
        plain=[];flipped=[];truth=[];names=[]
        with torch.inference_mode():
            for images,labels,batch_names in DataLoader(dataset,batch_size=1,num_workers=4,shuffle=False,pin_memory=True):
                images=images.cuda(non_blocking=True)
                with torch.autocast('cuda'):
                    a=probabilities_from_logits(model(images).flatten())
                    b=probabilities_from_logits(model(images.flip(-1)).flatten())
                plain.extend(a.cpu().tolist());flipped.extend(b.cpu().tolist())
                truth.extend(labels.tolist());names.extend(batch_names)
        if reference_names is not None and names!=reference_names:raise ValueError('Misaligned internal predictions')
        reference_names=names
        for tta in [False,True]:
            p=((np.array(plain,dtype=np.float32)+np.array(flipped,dtype=np.float32))/2).astype(np.float64) if tta else np.array(plain)
            candidates.append(dict(run=run,architecture=config['architecture'],epoch=saved['metrics']['epoch'],
                                   tta=tta,lr=config['lr'],batch=config['batch'],**metrics(truth,p)))
        del model;torch.cuda.empty_cache();print('Internal validation evaluated',run,flush=True)
    candidates.sort(key=lambda x:(x['auroc'],x['average_precision']),reverse=True)
    report=dict(schema_version=1,candidates=candidates,selected=candidates[0],manifest_sha256=fingerprint,evidence=evidence,
                source='official train holdout only; no official valid, test, leaderboard selection',
                patient_independence_verified=False)
    save_json(output/'internal_selection.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
