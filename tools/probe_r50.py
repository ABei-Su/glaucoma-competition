"""Synthetic real optimizer-path capacity check; never a trained candidate."""
import gc
import json
import sys
import time
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model import build_model
from utils import save_json


def probe(batch,memory_format):
    layout=torch.channels_last if memory_format=='channels_last' else torch.contiguous_format
    model=build_model(architecture='resnet50').to('cuda',memory_format=layout).train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda')
    x=torch.randn(batch,3,384,384,device='cuda').to(memory_format=layout)
    y=torch.arange(batch,device='cuda').remainder(2).float()
    torch.cuda.reset_peak_memory_stats()
    elapsed=[]
    for step in range(8):
        torch.cuda.synchronize();started=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x).flatten(),y)
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        scaler.step(optimizer);scaler.update()
        torch.cuda.synchronize()
        if step>=2: elapsed.append(time.perf_counter()-started)
    peak=torch.cuda.max_memory_allocated()
    result=dict(batch=batch,memory_format=memory_format,size=384,steps=6,mean_step_seconds=sum(elapsed)/len(elapsed),
                images_per_second=batch/(sum(elapsed)/len(elapsed)),peak_allocated_bytes=peak,
                status='passed' if peak<6_000_000_000 else 'over_memory_margin')
    return result


def main():
    torch.set_num_threads(4);torch.manual_seed(20261025)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    results=[]
    for memory_format in ['contiguous','channels_last']:
        for batch in [8,16,24]:
            try: item=probe(batch,memory_format)
            except torch.cuda.OutOfMemoryError as error:
                item=dict(batch=batch,memory_format=memory_format,size=384,status='oom',error=str(error))
            results.append(item);gc.collect();torch.cuda.empty_cache()
            print(json.dumps(item),flush=True)
    passing=[item for item in results if item['status']=='passed']
    if not passing: raise RuntimeError('ResNet50 384 capacity check failed')
    selected=max(passing,key=lambda item:item['images_per_second'])
    save_json(ROOT/'reports/r50_capacity.json',dict(gpu=torch.cuda.get_device_name(0),
              probes=results,selected_batch=selected['batch'],selected_memory_format=selected['memory_format'],source='synthetic inputs; no trained weights produced',
              indicative_compute_only_epoch_seconds=51396/selected['images_per_second']))
    print('SELECTED BATCH',selected['batch'],flush=True)


if __name__=='__main__': main()
