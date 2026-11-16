"""Bounded label-free latency diagnosis; never changes selection or predictions."""
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data import crop_fundus,pad_fundus,image_index,load_rows
from predict import Predictor
from provenance import inference_fingerprint
from utils import save_json,write_submission


def gpu_state():
    return subprocess.check_output(['nvidia-smi','--query-gpu=pstate,temperature.gpu,clocks.sm,power.draw,utilization.gpu','--format=csv,noheader'],text=True).strip()


def main():
    target=ROOT/'reports/v7_latency_diagnosis.json'
    if target.exists():raise FileExistsError('Keep the first bounded diagnosis; do not overwrite')
    torch.set_num_threads(4)
    config=ROOT/'output/v7/candidate_00.json'
    fingerprint=inference_fingerprint(config)
    raw=ROOT/'data/raw/glaucoma_first'
    rows=load_rows(raw,'valid')[:105]
    index=image_index(raw/'valid')
    paths=[index[row['Image_name'].casefold()] for row in rows]
    output=ROOT/'tmp/v7_latency_diagnosis';output.mkdir(exist_ok=True)
    report=dict(original_failed_mean_ms=108.90455633333335,files=fingerprint,arms=[],
                source='fixed official valid names only; labels unused',official_T4_verified=False)
    reference=None
    for deterministic in (False,True):
        torch.backends.cudnn.deterministic=deterministic
        torch.backends.cudnn.benchmark=False
        predictor=Predictor(config)
        for path in paths[:5]:predictor.predict_one(path)
        before=gpu_state();rounds=[];values=[]
        for repeat in range(3):
            durations=[]
            for path in paths[5:]:
                torch.cuda.synchronize();started=time.perf_counter()
                probability=predictor.predict_one(path)
                write_submission(output/(path.name+'.csv'),[path.name],[probability])
                torch.cuda.synchronize();durations.append((time.perf_counter()-started)*1000)
                values.append(probability)
            rounds.append(durations)
            print('arm',deterministic,'round',repeat+1,'mean_ms',sum(durations)/len(durations),flush=True)
        if reference is None:reference=values
        delta=max(abs(a-b) for a,b in zip(reference,values))
        arm=dict(deterministic=deterministic,benchmark=False,mean_ms=sum(map(sum,rounds))/300,
                 rounds_ms=rounds,max_abs_probability_delta_from_default=delta,gpu_before=before,gpu_after=gpu_state())
        report['arms'].append(arm);save_json(target,report)
        del predictor;torch.cuda.empty_cache()
    torch.backends.cudnn.deterministic=False
    predictor=Predictor(config)
    for path in paths[:5]:predictor.predict_one(path)
    stages=[]
    with torch.inference_mode():
        for path in paths[5:]:
            torch.cuda.synchronize();started=time.perf_counter()
            with Image.open(path) as image:cropped=crop_fundus(image)
            item=dict(read_crop_ms=(time.perf_counter()-started)*1000);tensors={};probabilities=[]
            for member,(model,size,tta) in enumerate(predictor.members):
                key=(size,model.inference_memory_format)
                if key not in tensors:
                    started=time.perf_counter()
                    layout=torch.channels_last if model.inference_memory_format=='channels_last' else torch.contiguous_format
                    tensors[key]=predictor.transform(pad_fundus(cropped,size)).unsqueeze(0).to(predictor.device,memory_format=layout)
                    torch.cuda.synchronize();item['prepare_'+str(member)+'_ms']=(time.perf_counter()-started)*1000
                started=time.perf_counter();tensor=tensors[key]
                from model import probabilities_from_logits
                with torch.autocast('cuda'):
                    probability=probabilities_from_logits(model(tensor).flatten())
                    if tta:probability=(probability+probabilities_from_logits(model(tensor.flip(-1)).flatten()))/2
                probabilities.append(probability.item())
                item['forward_'+str(member)+'_ms']=(time.perf_counter()-started)*1000
            started=time.perf_counter()
            write_submission(output/(path.name+'.profile.csv'),[path.name],[sum(probabilities)/len(probabilities)])
            item['write_ms']=(time.perf_counter()-started)*1000;stages.append(item)
    report['profile_ms']=stages
    report['profile_means_ms']={key:sum(item.get(key,0) for item in stages)/len(stages) for key in stages[0]}
    if inference_fingerprint(config)!=fingerprint:raise RuntimeError('Evidence changed during diagnosis')
    save_json(target,report)
    print(json.dumps(dict(arms=[{k:v for k,v in arm.items() if k!='rounds_ms'} for arm in report['arms']],
                          profile_means_ms=report['profile_means_ms'])),flush=True)


if __name__=='__main__':main()
