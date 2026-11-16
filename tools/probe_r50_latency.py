"""Actual batch1 raw-image inference timing on validation images before fitting."""
import json
import sys
import time
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data import load_rows,image_index
from model import build_model
from predict import Predictor
from utils import save_json,write_submission


def main():
    torch.set_num_threads(4)
    model=build_model(ROOT/'pretrain_model/resnet50-11ad3fa6.pth',architecture='resnet50')
    capacity=json.loads((ROOT/'reports/r50_capacity.json').read_text())
    memory_format=capacity.get('selected_memory_format','contiguous')
    checkpoint=ROOT/'tmp/r50_latency_only.pt'
    torch.save(dict(model=model.state_dict(),architecture='resnet50',size=384,memory_format=memory_format),checkpoint)
    del model
    predictor=Predictor(checkpoint)
    raw=ROOT/'data/raw/glaucoma_first'
    rows=load_rows(raw,'valid');paths=image_index(raw/'valid')
    for row in rows[:5]: predictor.predict_one(paths[row['Image_name'].casefold()])
    times=[]
    for repeat in range(3):
        for row in rows[5:105]:
            torch.cuda.synchronize();start=time.perf_counter()
            probability=predictor.predict_one(paths[row['Image_name'].casefold()])
            write_submission(ROOT/'tmp/r50_probe_result.csv',[row['Image_name']],[probability])
            torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
    report=dict(mean_ms=sum(times)/len(times),memory_format=memory_format,batch=1,samples=100,rounds=3,warmup=5,
                gpu=torch.cuda.get_device_name(0),checkpoint_bytes=checkpoint.stat().st_size,
                note='Pretrained backbone plus random binary head; timing only, not a trained candidate or clinical result',
                source='official validation images',official_T4_verified=False)
    save_json(ROOT/'reports/r50_latency_probe.json',report)
    if report['mean_ms']>100: raise RuntimeError('Raw-image path exceeds local 100ms; investigate implementation before training')
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
