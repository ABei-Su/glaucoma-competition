"""One fixed resource/equivalence trial of the explicitly fused v7 deployment."""
import json
import sys
import time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data import image_index,load_rows
from predict import Predictor
from provenance import inference_fingerprint
from utils import save_json,write_submission


def main():
    target=ROOT/'reports/v7_fusion_probe.json'
    if target.exists():raise FileExistsError('Preserve the first fixed fusion trial')
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
    original=ROOT/'output/v7/candidate_00.json'
    fingerprint=inference_fingerprint(original)
    config=json.loads(original.read_text())
    config['models'][0]['fuse_conv_bn']=True
    config['name']+='_conv_bn_fused'
    config['deployment_revision']='ADR-006; inference-only Conv/BN folding, original checkpoints unchanged'
    candidate=original.with_name('fused_probe_config.json')
    if candidate.exists():raise FileExistsError('Preserve existing fusion candidate')
    save_json(candidate,config)
    raw=ROOT/'data/raw/glaucoma_first';index=image_index(raw/'valid')
    names=[row['Image_name'] for row in load_rows(raw,'valid')[:105]]
    predictor=Predictor(original)
    for name in names[:5]:predictor.predict_one(index[name.casefold()])
    reference={name:predictor.predict_one(index[name.casefold()]) for name in names[5:]}
    del predictor;torch.cuda.empty_cache()
    predictor=Predictor(candidate)
    for name in names[:5]:predictor.predict_one(index[name.casefold()])
    directory=ROOT/'tmp/v7_fusion_timing';directory.mkdir(exist_ok=True)
    rounds=[];max_delta=0.;repeated={};repeat_delta=0.
    for repeat in range(3):
        durations=[]
        for name in names[5:]:
            torch.cuda.synchronize();started=time.perf_counter()
            probability=predictor.predict_one(index[name.casefold()])
            write_submission(directory/(name+'.csv'),[name],[probability])
            torch.cuda.synchronize();durations.append((time.perf_counter()-started)*1000)
            max_delta=max(max_delta,abs(probability-reference[name]))
            if name in repeated:repeat_delta=max(repeat_delta,abs(probability-repeated[name]))
            repeated[name]=probability
        rounds.append(durations);print('fused round',repeat+1,'mean_ms',sum(durations)/100,flush=True)
    mean=sum(map(sum,rounds))/300
    report=dict(mean_ms=mean,rounds_ms=rounds,max_abs_probability_delta=max_delta,
                max_repeated_probability_delta=repeat_delta,batch=1,samples=100,warmup=5,rounds=3,
                within_limit=mean<=100,numeric_tolerance=.002,numeric_equivalence_passed=max_delta<=.002 and repeat_delta<=1e-7,
                source='fixed official valid names; labels unused',original_files=fingerprint,
                fused_files=inference_fingerprint(candidate),official_T4_verified=False)
    if inference_fingerprint(original)!=fingerprint:raise RuntimeError('Original evidence changed')
    save_json(target,report)
    print(json.dumps({key:value for key,value in report.items() if key not in ('rounds_ms','original_files','fused_files')}),flush=True)
    if not report['within_limit'] or not report['numeric_equivalence_passed']:raise SystemExit(1)


if __name__=='__main__':main()
