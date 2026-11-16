"""One bounded fixed-focal experiment; unchanged legacy trainer."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data import FundusDataset,audit_inputs,load_rows,image_index
from internal_split import partition_rows
from model import build_model
from focal_loss import FocalBinaryLoss
from predict import Predictor
from provenance import digest,inference_fingerprint
from utils import metrics,save_json,write_submission
from v8_integrity import control
from v10_integrity import verify_reference_probabilities
from tools.compare_v10_internal import evaluate
from tools.predict_v12 import configure_backend
from v12_protocol import recipe,training_command,transition,resource_gate,quality_gate,verify_evidence,frozen,validate_completed,inference_identity,RUNTIME


def bundle(path,checkpoint):
    members=[ROOT/'runs/v3_r50_384_w1/best.pt',Path(checkpoint).resolve()]
    save_json(path,dict(schema_version=1,name='v12_fixed_equal_r50_384_cn384_focal_tta',
        models=[dict(checkpoint=str(p),sha256=digest(p),tta=True) for p in members],
        batch=1,precision='fp16',probability_precision='fp32',threshold=.5,preprocess_version=1))
    return inference_identity(path)


def runtime_bytes(config):
    spec=json.loads(Path(config).read_text(encoding='utf-8'))
    return (sum((Path(config).parent/m['checkpoint']).stat().st_size for m in spec['models'])+
            Path(config).stat().st_size+sum((ROOT/name).stat().st_size for name in RUNTIME))


def measure(config,label):
    configure_backend();identity=inference_identity(config);predictor=Predictor(config)
    raw=ROOT/'data/raw/glaucoma_first';paths=image_index(raw/'valid');rows=load_rows(raw,'valid')[:105]
    for row in rows[:5]:predictor.predict_one(paths[row['Image_name'].casefold()])
    directory=ROOT/'tmp'/label;directory.mkdir(exist_ok=False);rounds=[];previous={};delta=0.
    for repeat in range(3):
        durations=[]
        for row in rows[5:]:
            name=row['Image_name'];torch.cuda.synchronize();started=time.perf_counter()
            p=predictor.predict_one(paths[name.casefold()]);write_submission(directory/(name+'.csv'),[name],[p])
            torch.cuda.synchronize();durations.append((time.perf_counter()-started)*1000)
            if name in previous:delta=max(delta,abs(p-previous[name]))
            previous[name]=p
        rounds.append(durations)
    if identity!=inference_identity(config):raise ValueError('Timing inputs changed')
    result=dict(mean_ms=sum(map(sum,rounds))/300,rounds_ms=rounds,max_repeat_delta=delta,
                runtime_bytes=runtime_bytes(config),batch=1,samples=100,warmup=5,rounds=3,official_T4_verified=False)
    del predictor;torch.cuda.empty_cache();return result


def probe(output):
    configure_backend();torch.manual_seed(20261031)
    model=build_model(architecture='convnext_tiny').cuda().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001)
    scaler=torch.amp.GradScaler('cuda');images=torch.randn(16,3,384,384,device='cuda')
    labels=torch.arange(16,device='cuda').remainder(2).float();durations=[];torch.cuda.reset_peak_memory_stats()
    for step in range(6):
        torch.cuda.synchronize();started=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):loss=FocalBinaryLoss()(model(images).flatten(),labels)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite synthetic resource loss')
        scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
        if step>=2:durations.append(time.perf_counter()-started)
    peak=torch.cuda.max_memory_allocated()
    capacity=dict(batch=16,size=384,steps=6,peak_allocated_bytes=peak,images_per_second=16/(sum(durations)/len(durations)))
    del model,optimizer,scaler,images,labels,loss;torch.cuda.empty_cache()
    if peak>=5_000_000_000:return dict(passed=False,capacity=capacity,reason='Fixed batch16 capacity gate failed')
    torch.manual_seed(20261031);model=build_model(ROOT/'pretrain_model/convnext_tiny-983f1562.pth',architecture='convnext_tiny')
    checkpoint=output/'resource_only_cn384.pt'
    torch.save(dict(model=model.state_dict(),architecture='convnext_tiny',size=384,memory_format='contiguous',
                    schema_version=1,resource_probe_only=True,validation_probability_precision='fp32'),checkpoint)
    del model;config=output/'resource_config.json';identity=bundle(config,checkpoint)
    timing=measure(config,'v12_resource_timing')
    return dict(passed=resource_gate(peak,timing['mean_ms'],timing['max_repeat_delta'],timing['runtime_bytes']),
                capacity=capacity,timing=timing,evidence=identity)


def run(command,log):
    with (ROOT/'reports'/log).open('x',encoding='utf-8') as stream:
        subprocess.run([sys.executable,'-u']+command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)


def compare_internal():
    record=frozen(ROOT);output=ROOT/'output/v12';raw=ROOT/'data/raw/glaucoma_first'
    if (output/'internal_selection.json').exists():raise FileExistsError('Preserve v12 comparison')
    audits={size:audit_inputs(raw,ROOT/('data/cache'+str(size)),size) for size in (384,)}
    _,held,split_sha=partition_rows(load_rows(raw,'train'),(raw/'train.csv').read_bytes(),ROOT/'reports/v5_internal_split.json')
    old=control(ROOT);new=validate_completed(ROOT,True);configure_backend();evidence=list(record['evidence'])
    with np.load(ROOT/'output/v8/internal_predictions.npz',allow_pickle=False) as archive:
        expected_names=archive['names'].tolist();expected_labels=archive['labels'].tolist()
        expected_probabilities=archive['v5_inner_convnext_clean_tta'].tolist()
    results={};arrays={};reference_delta=None
    for label,checked,size in [('reference',old,384),('candidate',new,384)]:
        data=FundusDataset(raw,'train',size,ROOT/('data/cache'+str(size)));data.rows=held
        names,labels,p=evaluate(checked['checkpoint'],data,False)
        if names!=expected_names or labels!=expected_labels:raise ValueError('Fixed-focal comparison misaligned')
        if label=='reference':reference_delta=verify_reference_probabilities(p,expected_probabilities)
        results[label]=metrics(labels,p);arrays[label]=np.asarray(p)
        print(label,json.dumps(results[label]),flush=True)
    for name in ('best.pt','config.json','best_metrics.json','status.json','history.jsonl','split_binding.json','environment.json'):
        path=ROOT/'runs/v12_inner_focal'/name;evidence.append(dict(path=str(path),sha256=digest(path)))
    path=output/'internal_predictions.npz'
    with path.open('xb') as stream:np.savez_compressed(stream,names=np.array(names),labels=np.array(labels),**arrays)
    evidence.append(dict(path=str(path),sha256=digest(path)))
    for size in (384,):
        if audit_inputs(raw,ROOT/('data/cache'+str(size)),size)!=audits[size]:raise ValueError('Comparison data audit changed')
    verify_evidence(evidence)
    report=dict(metrics=results,gate=quality_gate(results['reference'],results['candidate']),
        selected_epoch=new['checkpoint']['metrics']['epoch'],manifest_sha256=split_sha,evidence=evidence,
        rows=len(labels),positive_rows=int(sum(labels)),reference_max_probability_delta=reference_delta,
        reference_probability_tolerance=1e-7,source='reused officialtrain holdout only',
        limitations='Exploratory image-level comparison; patient groups unavailable; not leaderboard or clinical performance')
    save_json(output/'internal_selection.json',report);print('V12 INTERNAL GATE',json.dumps(report['gate']),flush=True)
    return report


def main():
    output=ROOT/'output/v12';status=ROOT/'reports/v12_status.json'
    if status.exists() or output.exists() or any((ROOT/'runs'/name).exists() for name in ('v12_inner_focal','v12_full_focal')):
        raise FileExistsError('Preserve existing v12 operation; no automatic recovery')
    output.mkdir(exist_ok=False);lock=output/'operation.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    state=dict(kind='experiment',phase='ready',revision=0,history=[])
    def advance(phase,**details):
        nonlocal state
        state=transition(state,phase,details);save_json(status,state)
    try:
        advance('preflight');log=ROOT/'reports/v12_verification.log'
        if 'VERIFY PASSED:' not in log.read_text(encoding='utf-8'):raise ValueError('V12 verification required')
        control(ROOT)
        cache=json.loads((ROOT/'data/cache384/READY.json').read_text(encoding='utf-8'))
        if cache['manifest_sha256']!='cba551a26d60f79398538b7e7edb3e60f9ff77811d119120f01e58972c7bb409' or cache['size']!=384:
            raise ValueError('Unexpected384cache')
        paths=list(ROOT.glob('*.py'))+list((ROOT/'tools').glob('*.py'))+list((ROOT/'tests').glob('*.py'))
        paths += [ROOT/name for name in ('docs/OPTIMIZATION-v12.md','docs/ADR-012-focal-training.md',
            'docs/science-superpowers/questions/2026-10-28-focal.md','docs/REVIEW-v12-preflight.md',
            'reports/v12_verification.log','data/cache384/READY.json','data/cache384/READY.json',
            'reports/v5_internal_split.json','data/raw/glaucoma_first/train.csv','data/raw/glaucoma_first/valid.csv',
            'tmp/derive_v12.py','reports/v12_derivation.json','pretrain_model/convnext_tiny-983f1562.pth','output/v8/internal_predictions.npz','output/v8/internal_selection.json')]
        paths += [ROOT/'runs/v5_inner_convnext'/name for name in ('best.pt','config.json','best_metrics.json','status.json')]
        evidence=[dict(path=str(path),sha256=digest(path)) for path in sorted(set(paths))]
        evidence+=inference_fingerprint(ROOT/'output/v6/selected_config.json')
        save_json(output/'preflight_inputs.json',dict(evidence=evidence,internal_recipe=recipe(True),full_seed=20261032,
            environment=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0))))
        resource=probe(output);verify_evidence(evidence)
        save_json(ROOT/'reports/v12_resource_probe.json',resource)
        if not resource['passed']:advance('not_promoted',reason='Fixed resource gate failed',resource=resource);return
        evidence+=resource['evidence']+[dict(path=str(ROOT/'reports/v12_resource_probe.json'),sha256=digest(ROOT/'reports/v12_resource_probe.json'))]
        save_json(output/'frozen_inputs.json',dict(evidence=evidence,internal_recipe=recipe(True),full_seed=20261032,
            trial_csv_policy='Unique qualified full candidate saved after resource/identity checks; validation regression disclosed'))
        frozen(ROOT);advance('internal_training',resource_passed=True,run='v12_inner_focal')
        run(training_command(True),'v12_inner_focal.log');validate_completed(ROOT,True);frozen(ROOT)
        advance('internal_comparison');selection=compare_internal();frozen(ROOT)
        if not selection['gate']['passed']:
            advance('not_promoted',reason='Internal384AUROC/AP gate failed',gate=selection['gate']);return
        epoch=selection['selected_epoch'];advance('full_training',gate=selection['gate'],frozen_epoch=epoch,run='v12_full_focal')
        run(training_command(False,epoch),'v12_full_focal.log');validate_completed(ROOT,False,epoch);frozen(ROOT)
        advance('full_ready',frozen_epoch=epoch,gate=selection['gate']);print('V12 FULL READY',flush=True)
    except BaseException as error:
        if state['phase'] not in ('failed','not_promoted','full_ready'):advance('failed',error=repr(error))
        raise
    finally:
        if lock.exists() and lock.read_text().strip()==str(os.getpid()):lock.unlink()


if __name__=='__main__':main()
