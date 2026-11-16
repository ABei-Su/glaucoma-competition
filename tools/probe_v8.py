"""One label-free timing campaign for the fixed R50/CN inference shapes."""
import json
import sys
import time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data import load_rows,image_index
from predict import Predictor
from provenance import inference_fingerprint,digest
from utils import save_json,write_submission


def main():
    target=ROOT/'reports/v8_latency_probe.json'
    if target.exists():raise FileExistsError('Preserve the first timing campaign')
    torch.set_num_threads(4)
    output=ROOT/'output/v8';output.mkdir(exist_ok=True)
    source=ROOT/'output/v6/selected_config.json';spec=json.loads(source.read_text())
    members=[dict(checkpoint=str((source.parent/item['checkpoint']).resolve()),tta=True) for item in spec['models']]
    if len(members)!=2:raise ValueError('Expected fixed two-member v6 reference')
    for member,architecture in zip(members,['resnet50','convnext_tiny']):
        if json.loads((Path(member['checkpoint']).parent/'config.json').read_text())['architecture']!=architecture:
            raise ValueError('Reference member order/architecture changed')
    evidence=inference_fingerprint(source)+[dict(path=str(ROOT/name),sha256=digest(ROOT/name)) for name in ['predict.py','model.py','data.py']]
    raw=ROOT/'data/raw/glaucoma_first';rows=load_rows(raw,'valid')[:105];paths=image_index(raw/'valid')
    directory=ROOT/'tmp/v8_resource_timing';directory.mkdir(exist_ok=True)
    timings={};eligible={}
    for mode in ['plain','tta']:
        config=output/('probe_'+mode+'.json')
        entries=[dict(members[0]),dict(members[1],tta=mode=='tta')]
        save_json(config,dict(schema_version=1,models=entries,name='v8_resource_'+mode))
        predictor=Predictor(config)
        for row in rows[:5]:predictor.predict_one(paths[row['Image_name'].casefold()])
        rounds=[];reference={};delta=0.
        for repeat in range(3):
            times=[]
            for row in rows[5:]:
                name=row['Image_name'];torch.cuda.synchronize();start=time.perf_counter()
                p=predictor.predict_one(paths[name.casefold()]);write_submission(directory/(name+'.csv'),[name],[p])
                torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
                if name in reference:delta=max(delta,abs(p-reference[name]))
                reference[name]=p
            rounds.append(times)
        mean=sum(map(sum,rounds))/300
        if delta>1e-7:raise ValueError('Nondeterministic resource probe')
        timings[mode]=dict(mean_ms=mean,rounds_ms=rounds,max_repeat_delta=delta,batch=1,samples=100,warmup=5,rounds=3)
        eligible[mode]=mean<=90
        del predictor;torch.cuda.empty_cache();print(mode,mean,flush=True)
    save_json(target,dict(timings=timings,eligible=eligible,local_initial_margin_ms=90,final_rule_ms=100,
              evidence=evidence,official_T4_verified=False,source='fixed official valid names only; labels unused'))


if __name__=='__main__':main()
