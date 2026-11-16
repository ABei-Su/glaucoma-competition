"""Synthetic capacity and four fixed latency shapes, no outcome-based decisions."""
import gc
import json
import sys
import time
from pathlib import Path
import torch
from probe_v5 import capacity
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model import build_model
from predict import Predictor
from data import image_index,load_rows
from utils import save_json,write_submission
from v7_protocol import MODES


def main():
    torch.set_num_threads(4);torch.manual_seed(20261035)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    probes=[]
    for batch in [4,8,12,16]:
        try:item=capacity('efficientnet_v2_s',batch)
        except torch.cuda.OutOfMemoryError as error:item=dict(batch=batch,status='oom',error=str(error))
        probes.append(item);gc.collect();torch.cuda.empty_cache();print(json.dumps(item),flush=True)
    good=[item for item in probes if item['status']=='passed']
    if not good:raise RuntimeError('No EfficientNet training batch passed')
    save_json(ROOT/'reports/v7_capacity.json',dict(probes=probes,selected_batch=max(good,key=lambda x:x['images_per_second'])['batch'],
                                                size=384,memory_format='contiguous',gpu=torch.cuda.get_device_name(0)))
    model=build_model(architecture='efficientnet_v2_s')
    checkpoint=ROOT/'tmp/v7_latency_only.pt';torch.save(dict(model=model.state_dict(),architecture='efficientnet_v2_s',size=384),checkpoint)
    del model
    raw=ROOT/'data/raw/glaucoma_first';rows=load_rows(raw,'valid');paths=image_index(raw/'valid')
    timings={};eligible={}
    for mode in MODES:
        tta=mode.endswith('_tta');members=[dict(checkpoint=checkpoint.name,tta=tta)]
        if mode.startswith('pair_'):members.append(dict(checkpoint='../runs/v5_full/best.pt',tta=tta))
        config=ROOT/'tmp'/('v7_latency_'+mode+'.json');save_json(config,dict(schema_version=1,models=members))
        predictor=Predictor(config)
        for row in rows[:5]:predictor.predict_one(paths[row['Image_name'].casefold()])
        elapsed=[]
        for repeat in range(3):
            for row in rows[5:105]:
                torch.cuda.synchronize();started=time.perf_counter()
                p=predictor.predict_one(paths[row['Image_name'].casefold()])
                write_submission(ROOT/'tmp/v7_latency_result.csv',[row['Image_name']],[p])
                torch.cuda.synchronize();elapsed.append((time.perf_counter()-started)*1000)
        mean=sum(elapsed)/len(elapsed);timings[mode]=dict(mean_ms=mean,batch=1,samples=100,warmup=5,rounds=3)
        eligible[mode]=mean<=100
        del predictor;gc.collect();torch.cuda.empty_cache();print(mode,json.dumps(timings[mode]),flush=True)
    report=dict(timings=timings,eligible=eligible,source='official validation names; untrained EfficientNet model; resource measurements only',
                official_T4_verified=False,gpu=torch.cuda.get_device_name(0))
    save_json(ROOT/'reports/v7_latency_probe.json',report)
    if not any(eligible.values()):raise RuntimeError('No deployment shape passed measured latency')


if __name__=='__main__':main()
