"""Measured 448 capacity and actual raw-image latency before supervised fitting."""
import gc
import argparse
import json
import sys
import time
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model import build_model
from predict import Predictor
from data import image_index,load_rows
from utils import save_json,write_submission


def capacity(batch):
    model=build_model(architecture='resnet50').cuda().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda')
    x=torch.randn(batch,3,448,448,device='cuda'); y=torch.arange(batch,device='cuda').remainder(2).float()
    torch.cuda.reset_peak_memory_stats();elapsed=[]
    for step in range(8):
        torch.cuda.synchronize();started=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x).flatten(),y)
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.);scaler.step(optimizer);scaler.update()
        torch.cuda.synchronize()
        if step>=2: elapsed.append(time.perf_counter()-started)
    peak=torch.cuda.max_memory_allocated()
    return dict(batch=batch,size=448,memory_format='contiguous',peak_allocated_bytes=peak,
                images_per_second=batch/(sum(elapsed)/len(elapsed)),status='passed' if peak<6_000_000_000 else 'over_memory_margin')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--latency-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(20261027)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    if not args.latency_only:
        results=[]
        for batch in [8,12,16]:
            try: item=capacity(batch)
            except torch.cuda.OutOfMemoryError as error:
                item=dict(batch=batch,size=448,status='oom',error=str(error))
            results.append(item);gc.collect();torch.cuda.empty_cache();print(json.dumps(item),flush=True)
        good=[item for item in results if item['status']=='passed']
        if not good: raise RuntimeError('448 capacity failed')
        selected=max(good,key=lambda item:item['images_per_second'])
        save_json(ROOT/'reports/v4_capacity.json',dict(probes=results,selected_batch=selected['batch'],
                  gpu=torch.cuda.get_device_name(0),source='synthetic forward/backward/optimizer only'))
    source=ROOT/'runs/v3_r50_384_w1/best.pt'
    saved=torch.load(source,map_location='cpu',weights_only=True);saved['size']=448
    checkpoint=ROOT/'tmp/v4_latency_only.pt';torch.save(saved,checkpoint);del saved
    predictor=Predictor(checkpoint)
    raw=ROOT/'data/raw/glaucoma_first';rows=load_rows(raw,'valid');paths=image_index(raw/'valid')
    for row in rows[:5]: predictor.predict_one(paths[row['Image_name'].casefold()])
    times=[]
    for repeat in range(3):
        for row in rows[5:105]:
            torch.cuda.synchronize();started=time.perf_counter()
            probability=predictor.predict_one(paths[row['Image_name'].casefold()])
            write_submission(ROOT/'tmp/v4_latency_result.csv',[row['Image_name']],[probability])
            torch.cuda.synchronize();times.append((time.perf_counter()-started)*1000)
    report=dict(mean_ms=sum(times)/len(times),size=448,batch=1,samples=100,warmup=5,rounds=3,
                source='official validation images; size override for latency only, not a trained 448 candidate',
                official_T4_verified=False,gpu=torch.cuda.get_device_name(0))
    save_json(ROOT/'reports/v4_latency_probe.json',report);print(json.dumps(report),flush=True)
    if report['mean_ms']>100: raise RuntimeError('448 actual path exceeds local 100ms; investigate before fitting')


if __name__=='__main__': main()
