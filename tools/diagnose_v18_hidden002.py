"""Single-use instrumented VALID-only diagnosis; never an acceptance benchmark."""
import json
import os
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/v18/latency_diagnosis_hidden002'


def main():
    os.chdir(ROOT)
    OUT.mkdir(exist_ok=False)  # persistent one-shot reservation
    import torch
    import predict
    from data import image_index, load_rows
    from provenance import digest
    from utils import save_json, write_submission

    config = ROOT / 'output/v18/finalization/pair_config.json'
    raw = ROOT / 'data/raw/glaucoma_first'
    names = [r['Image_name'] for r in load_rows(raw, 'valid')]
    index = image_index(raw / 'valid')
    selected, warmup = names[:10], names[100:105]
    bound = json.loads((ROOT/'output/v18/finalization/frozen_trial_inputs.json').read_text('utf-8'))['files']
    known = {str(Path(x['path']).resolve()).casefold(): x['sha256'] for x in bound}
    spec = json.loads(config.read_text('utf-8'))
    files = [config, ROOT/'predict.py', ROOT/'data.py', ROOT/'model.py', ROOT/'utils.py']
    files += [Path(x['checkpoint']) for x in spec['models']]
    files += [index[n.casefold()] for n in selected + warmup]
    def verify():
        for p in files:
            if digest(p) != known[str(p.resolve()).casefold()]:
                raise ValueError('Frozen diagnostic input changed: '+str(p))
    verify()
    report = dict(purpose='Component diagnosis only; not resource pass/fail or official timing',
                  sample='first10 officialvalid; warmup rows100:105; one instrumented pass',
                  timed_names=selected, warmup_names=warmup, started_at=time.time(),
                  perturbations='Extra CUDA synchronizations and background telemetry; totals not acceptance measurements',
                  state='started', records=[], telemetry=[], files=[dict(path=str(p),sha256=digest(p)) for p in files])
    save_json(OUT/'report.json', report)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    report['environment'] = dict(torch=torch.__version__,cuda=torch.version.cuda,
                                gpu=torch.cuda.get_device_name(0),threads=torch.get_num_threads())
    predictor = predict.Predictor(config)
    for n in warmup:
        predictor.predict_one(index[n.casefold()])
    torch.cuda.synchronize()
    stop = threading.Event()
    def telemetry():
        while not stop.is_set():
            try:
                p = subprocess.run(['nvidia-smi', '--query-gpu=timestamp,pstate,temperature.gpu,utilization.gpu,clocks.current.graphics,clocks.current.memory,power.draw', '--format=csv,noheader'],
                                   capture_output=True,text=True,timeout=5,creationflags=0x08000000)
                report['telemetry'].append(dict(time=time.time(),returncode=p.returncode,output=p.stdout.strip(),error=p.stderr.strip()))
            except Exception as exc:
                report['telemetry'].append(dict(time=time.time(),error=repr(exc)))
            stop.wait(.5)
    thread=threading.Thread(target=telemetry,daemon=True)
    current={}
    def wrapped(fn,key,sync=False):
        def call(*args,**kwargs):
            if sync:torch.cuda.synchronize()
            t=time.perf_counter()
            result=fn(*args,**kwargs)
            if sync:torch.cuda.synchronize()
            current[key]=current.get(key,0.)+(time.perf_counter()-t)*1000
            return result
        return call
    original_crop,original_pad = predict.crop_fundus,predict.pad_fundus
    original_transform = predictor.transform
    original_forwards = [m.forward for m,_,_ in predictor.members]
    predict.crop_fundus=wrapped(original_crop,'decode_and_crop_ms')
    predict.pad_fundus=wrapped(original_pad,'resize_pad_ms')
    predictor.transform=wrapped(original_transform,'tensor_normalize_ms')
    for i,(m,_,_) in enumerate(predictor.members):
        m.forward=wrapped(original_forwards[i],f'model_{i}_two_forwards_ms',sync=True)
    thread.start()
    try:
        for n in selected:
            current.clear()
            torch.cuda.synchronize();start=time.perf_counter()
            value=predictor.predict_one(index[n.casefold()])
            torch.cuda.synchronize()
            inference_ms=(time.perf_counter()-start)*1000
            start=time.perf_counter()
            write_submission(OUT/'diagnostic_row.csv',[n],[value])
            write_ms=(time.perf_counter()-start)*1000
            report['records'].append(dict(name=n,components=dict(current),inference_ms=inference_ms,
                write_ms=write_ms,unattributed_ms=inference_ms-sum(current.values())))
        report['state']='complete'
    except BaseException as exc:
        report.update(state='failed',error=repr(exc))
        raise
    finally:
        stop.set();thread.join(timeout=6)
        predict.crop_fundus,predict.pad_fundus=original_crop,original_pad
        predictor.transform=original_transform
        for (m,_,_),forward in zip(predictor.members,original_forwards):m.forward=forward
        report['finished_at']=time.time()
        save_json(OUT/'report.json',report)
    verify()
    report['input_hashes_unchanged']=True
    save_json(OUT/'report.json',report)
    print(json.dumps(dict(state=report['state'],observations=len(report['records']),report=str(OUT/'report.json'))))


if __name__=='__main__':main()
