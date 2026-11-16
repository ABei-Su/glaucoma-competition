"""Bounded hardware probe, using synthetic inputs only."""
import json
import sys
import time
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model import build_model
from utils import save_json


def main():
    torch.set_num_threads(4)
    torch.manual_seed(7)
    model=build_model().cuda()
    images=torch.randn(48,3,224,224,device='cuda')
    labels=torch.rand(48,device='cuda')
    optimizer=torch.optim.AdamW(model.parameters())
    scaler=torch.amp.GradScaler('cuda')
    started=time.perf_counter()
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda'):
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(images).flatten(),labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    result={'architecture':'resnet18','status':'passed','loss':float(loss),
            'seconds_12_steps':time.perf_counter()-started,
            'peak_allocated_GB':torch.cuda.max_memory_allocated()/1e9,'batch':48}
    if not torch.isfinite(loss):
        raise FloatingPointError('GPU smoke loss not finite')
    save_json(ROOT/'reports'/'gpu_smoke.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    main()
