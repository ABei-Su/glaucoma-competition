"""Synthetic training capacity and label-free latency; no outcome selection."""
import gc
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


def capacity(architecture,batch):
    model=build_model(architecture=architecture).cuda().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    scaler=torch.amp.GradScaler('cuda')
    x=torch.randn(batch,3,384,384,device='cuda');y=torch.arange(batch,device='cuda').remainder(2).float()
    torch.cuda.reset_peak_memory_stats();elapsed=[]
    for step in range(6):
        torch.cuda.synchronize();started=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x).flatten(),y)
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.);scaler.step(optimizer);scaler.update()
        torch.cuda.synchronize()
        if step>=2:elapsed.append(time.perf_counter()-started)
    peak=torch.cuda.max_memory_allocated()
    return dict(batch=batch,peak_allocated_bytes=peak,images_per_second=batch/(sum(elapsed)/len(elapsed)),
                status='passed' if peak<5_000_000_000 else 'over_memory_margin')


def main():
    torch.set_num_threads(4);torch.manual_seed(20261029)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    report=dict(gpu=torch.cuda.get_device_name(0),size=384,memory_format='contiguous',architectures={})
    for architecture in ['convnext_tiny','resnet50']:
        probes=[]
        for batch in [8,12,16]:
            try:item=capacity(architecture,batch)
            except torch.cuda.OutOfMemoryError as error:item=dict(batch=batch,status='oom',error=str(error))
            probes.append(item);gc.collect();torch.cuda.empty_cache();print(architecture,json.dumps(item),flush=True)
        good=[item for item in probes if item['status']=='passed']
        if not good:raise RuntimeError('No batch passed for '+architecture)
        report['architectures'][architecture]=dict(probes=probes,selected_batch=max(good,key=lambda p:p['images_per_second'])['batch'])
        save_json(ROOT/'reports/v5_capacity.json',report)
    model=build_model(architecture='convnext_tiny')
    checkpoint=ROOT/'tmp/v5_latency_only.pt'
    torch.save(dict(model=model.state_dict(),architecture='convnext_tiny',size=384),checkpoint)
    del model
    config=ROOT/'tmp/v5_latency_config.json'
    save_json(config,dict(schema_version=1,models=[dict(checkpoint=checkpoint.name,tta=True)]))
    predictor=Predictor(config)
    raw=ROOT/'data/raw/glaucoma_first';paths=image_index(raw/'valid');rows=load_rows(raw,'valid')
    for row in rows[:5]:predictor.predict_one(paths[row['Image_name'].casefold()])
    elapsed=[]
    for repeat in range(3):
        for row in rows[5:105]:
            torch.cuda.synchronize();started=time.perf_counter()
            p=predictor.predict_one(paths[row['Image_name'].casefold()])
            write_submission(ROOT/'tmp/v5_latency_result.csv',[row['Image_name']],[p])
            torch.cuda.synchronize();elapsed.append((time.perf_counter()-started)*1000)
    latency=dict(mean_ms=sum(elapsed)/len(elapsed),batch=1,rounds=3,samples=100,warmup=5,tta=True,
                 source='official validation images; untrained model; resource probe only',
                 official_T4_verified=False,gpu=torch.cuda.get_device_name(0))
    save_json(ROOT/'reports/v5_latency_probe.json',latency);print(json.dumps(latency),flush=True)
    if latency['mean_ms']>100:raise RuntimeError('ConvNeXt local latency gate failed')


if __name__=='__main__':main()
