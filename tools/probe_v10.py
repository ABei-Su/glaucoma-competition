"""One synthetic training-capacity and label-free fixed-pair latency campaign."""
import json
import sys
import time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from clinical import build_age_model,fit_age_stats
from clinical_predict import Predictor,fingerprint,write_age_metadata
from data import load_rows,image_index
from internal_split import partition_rows
from utils import save_json,write_submission
from provenance import digest


def main():
    target=ROOT/'reports/v10_resource_probe.json'
    if target.exists():raise FileExistsError('Preserve first v10 resource campaign')
    parent=ROOT/'output/v10';parent.mkdir(exist_ok=True)
    output=parent/'resource_final';output.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(20261031)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    model=build_age_model().cuda().train()
    optimizer=torch.optim.AdamW([dict(params=model.image.parameters(),lr=.0001,weight_decay=.0001),
                                 dict(params=model.age.parameters(),lr=.001,weight_decay=.01)])
    scaler=torch.amp.GradScaler('cuda');x=torch.randn(16,3,384,384,device='cuda')
    ages=torch.zeros(16,2,device='cuda');ages[:,0]=torch.linspace(-1,1,16,device='cuda')
    labels=torch.arange(16,device='cuda').remainder(2).float();times=[]
    torch.cuda.reset_peak_memory_stats()
    for step in range(6):
        torch.cuda.synchronize();start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x,ages),labels)
        if not torch.isfinite(loss):raise ValueError('Synthetic Age capacity loss nonfinite')
        scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
        if step>=2:times.append(time.perf_counter()-start)
    peak=torch.cuda.max_memory_allocated();capacity=dict(batch=16,steps=6,peak_allocated_bytes=peak,images_per_second=16/(sum(times)/len(times)))
    del model,optimizer,scaler,x,ages,labels,loss;torch.cuda.empty_cache()
    if peak>=5_000_000_000:
        save_json(target,dict(passed=False,capacity=capacity,reason='5GB allocated memory margin exceeded'));return
    raw=ROOT/'data/raw/glaucoma_first'
    trained,_,_=partition_rows(load_rows(raw,'train'),(raw/'train.csv').read_bytes(),ROOT/'reports/v5_internal_split.json')
    stats=fit_age_stats(trained);public=ROOT/'pretrain_model/convnext_tiny-983f1562.pth'
    model=build_age_model(public)
    checkpoint=output/'resource_only_age.pt'
    if checkpoint.exists():raise FileExistsError('Preserve resource-only weights')
    torch.save(dict(schema_version=2,architecture='convnext_tiny_age',model=model.state_dict(),size=384,
                    age_stats=stats,memory_format='contiguous',resource_probe_only=True),checkpoint)
    del model
    metadata=output/'resource_valid_age.csv';write_age_metadata(raw/'valid.csv',metadata)
    config=output/'resource_config.json'
    r50=ROOT/'runs/v3_r50_384_w1/best.pt'
    save_json(config,dict(schema_version=2,age_metadata=metadata.name,age_metadata_sha256=digest(metadata),
        models=[dict(checkpoint=str(r50),sha256=digest(r50),tta=True),dict(checkpoint=checkpoint.name,sha256=digest(checkpoint),tta=True)]))
    evidence=fingerprint(config);predictor=Predictor(config);rows=load_rows(raw,'valid')[:105];paths=image_index(raw/'valid')
    for row in rows[:5]:predictor.predict_one(paths[row['Image_name'].casefold()])
    directory=ROOT/'tmp/v10_timing';directory.mkdir(exist_ok=True)
    rounds=[];reference={};delta=0.
    for repeat in range(3):
        durations=[]
        for row in rows[5:]:
            name=row['Image_name'];torch.cuda.synchronize();start=time.perf_counter()
            p=predictor.predict_one(paths[name.casefold()]);write_submission(directory/(name+'.csv'),[name],[p])
            torch.cuda.synchronize();durations.append((time.perf_counter()-start)*1000)
            if name in reference:delta=max(delta,abs(reference[name]-p))
            reference[name]=p
        rounds.append(durations)
    if fingerprint(config)!=evidence or delta>1e-7:raise ValueError('Resource inference changed or is nondeterministic')
    mean=sum(map(sum,rounds))/300
    report=dict(passed=mean<=100,capacity=capacity,mean_ms=mean,limit_ms=100,batch=1,samples=100,warmup=5,rounds=3,
                rounds_ms=rounds,max_repeat_delta=delta,evidence=evidence,source='Synthetic capacity; officialvalid filenames/images/Age inputs, labels unused',
                official_T4_verified=False)
    save_json(target,report);print(json.dumps({k:v for k,v in report.items() if k not in ('rounds_ms','evidence')}),flush=True)


if __name__=='__main__':main()
