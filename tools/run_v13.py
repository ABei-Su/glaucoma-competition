"""Bounded v13 campaign: one preflight, six paired fits, conditional full delivery."""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import FundusDataset, audit_inputs, image_index, load_rows
from model import build_model
from predict import Predictor, test_names
from provenance import digest, inference_fingerprint
from train_mixup import PUBLIC_SHA
from utils import metrics, save_json, write_submission
from v13_core import Mixup, promotion, soft_bce, validate_folds
from v13_protocol import TERMINAL, remaining_timeout, resource_gate, transition, stage_guard, require_promotion

OUT = ROOT/'output/v13'
RAW = ROOT/'data/raw/glaucoma_first'
RUNTIME = ['predict.py','model.py','data.py','utils.py','provenance.py','cache_paths.py','tools/run_v13.py',
           'train_mixup.py','v13_core.py','v13_protocol.py']


def backend():
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def verify_evidence(items):
    for item in items:
        if digest(item['path']) != item['sha256']: raise ValueError('Evidence changed: '+item['path'])


def frozen():
    record = json.loads((OUT/'frozen_inputs.json').read_text(encoding='utf-8'))
    verify_evidence(record['evidence'])
    return record


def identity(config):
    return inference_fingerprint(config)+[dict(path=str(ROOT/n), sha256=digest(ROOT/n)) for n in RUNTIME]


def bundle(path, checkpoint):
    members = [ROOT/'runs/v3_r50_384_w1/best.pt', checkpoint]
    save_json(path, dict(schema_version=1, name='v13_fixed_equal_r50_cn384_mixup_tta',
                        models=[dict(checkpoint=str(p.resolve()),sha256=digest(p),tta=True) for p in members],
                        batch=1,precision='fp16',probability_precision='fp32',threshold=.5,preprocess_version=1))


def measure(config, label):
    backend(); before = identity(config); predictor = Predictor(config)
    rows = load_rows(RAW,'valid')[:105]; paths = image_index(RAW/'valid')
    for row in rows[100:]: predictor.predict_one(paths[row['Image_name'].casefold()])
    directory = ROOT/'tmp'/label; directory.mkdir(exist_ok=False)
    rounds, previous, delta = [], {}, 0.
    for repeat in range(3):
        durations = []
        for row in rows[:100]:
            name = row['Image_name']; torch.cuda.synchronize(); started = time.perf_counter()
            p = predictor.predict_one(paths[name.casefold()])
            write_submission(directory/(name+'.csv'), [name], [p])
            torch.cuda.synchronize(); durations.append((time.perf_counter()-started)*1000)
            if name in previous: delta = max(delta, abs(p-previous[name]))
            previous[name] = p
        rounds.append(durations)
    if before != identity(config): raise ValueError('Timing identity changed')
    spec = json.loads(config.read_text(encoding='utf-8'))
    runtime_size = sum(Path(m['checkpoint']).stat().st_size for m in spec['models'])+config.stat().st_size
    result = dict(mean_ms=float(np.mean(rounds)), rounds_ms=rounds, max_repeat_delta=delta,
                  runtime_bytes=runtime_size, runtime_source_bytes=sum((ROOT/n).stat().st_size for n in RUNTIME),
                  batch=1,samples=100,warmup=5,rounds=3,official_T4_verified=False,
                  timed_names=[r['Image_name'] for r in rows[:100]], warmup_names=[r['Image_name'] for r in rows[100:]])
    del predictor; torch.cuda.empty_cache()
    return result


def preflight():
    frozen(); backend(); torch.manual_seed(20261101)
    audit_inputs(RAW,ROOT/'data/cache384',384)
    np.random.seed(20261101)
    import random
    random.seed(20261101)
    dataset=FundusDataset(RAW,'train',384,ROOT/'data/cache384',True,augmentation='mild')
    batch=[dataset[i] for i in range(16)]
    public = ROOT/'pretrain_model/convnext_tiny-983f1562.pth'
    if digest(public) != PUBLIC_SHA: raise ValueError('Wrong public weights')
    model = build_model(public, architecture='convnext_tiny').cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001,weight_decay=.0001)
    scaler = torch.amp.GradScaler('cuda'); mixer = Mixup(20261101)
    images = torch.stack([item[0] for item in batch]).cuda()
    labels = torch.stack([item[1] for item in batch]).cuda()
    torch.cuda.reset_peak_memory_stats(); durations = []
    for step in range(6):
        torch.cuda.synchronize(); started = time.perf_counter()
        x,y = mixer(images,labels); optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda'): logits = model(x).flatten()
        loss = soft_bce(logits,y)
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite probe loss')
        scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
        durations.append(time.perf_counter()-started)
    peak = torch.cuda.max_memory_allocated()
    del model, optimizer, scaler, images, labels, x, y, logits, loss
    torch.cuda.empty_cache()
    report = dict(passed=False, peak_allocated_bytes=peak, step_seconds=durations, steps=6,
                  training_names=[item[2] for item in batch], data_scope='first16 officialtrain rows; real cached augmented images')
    if peak < 5_000_000_000:
        torch.manual_seed(20261101)
        model = build_model(public,architecture='convnext_tiny')
        checkpoint = OUT/'resource_only_cn384.pt'
        torch.save(dict(model=model.state_dict(),schema_version=1,architecture='convnext_tiny',size=384,
                        memory_format='contiguous',validation_probability_precision='fp32',resource_probe_only=True),checkpoint)
        del model
        config = OUT/'resource_config.json'; bundle(config,checkpoint)
        report['timing'] = measure(config,'v13_resource_timing')
        t = report['timing'];report['passed'] = resource_gate(peak,t['mean_ms'],t['max_repeat_delta'],t['runtime_bytes'])
    frozen();save_json(ROOT/'reports/v13_resource_probe.json',report)


def completed(run, fold=None, treatment='mixup', full=False):
    status = json.loads((run/'status.json').read_text(encoding='utf-8'))
    config = json.loads((run/'config.json').read_text(encoding='utf-8'))
    binding = json.loads((run/'binding.json').read_text(encoding='utf-8'))
    if status['state'] != 'complete' or status['epoch'] != 8 or status['binding'] != binding:
        raise ValueError('Incomplete training: '+str(run))
    if (config['treatment'] != treatment or config['full'] != full or config['epochs'] != 8
            or config['checkpoint_epoch'] != 8 or binding['public_pretrain_sha256'] != PUBLIC_SHA
            or config['seed'] != (20261104 if full else 20261101+fold)
            or binding['fold'] != fold): raise ValueError('Training recipe mismatch')
    for name, sha in binding['source_identity'].items():
        if digest(ROOT/name) != sha: raise ValueError('Training sources changed')
    checkpoint = torch.load(run/'best.pt',map_location='cpu',weights_only=True)
    if checkpoint['binding'] != binding or checkpoint['checkpoint_epoch'] != 8:
        raise ValueError('Checkpoint binding mismatch')
    return binding


def compare():
    frozen();manifest = json.loads((ROOT/'output/v13_planning/three_fold_manifest.json').read_text(encoding='utf-8'))
    rows = load_rows(RAW,'train'); labels = {r['Image_name']:int(r['True_Label']) for r in rows}
    all_values = {t:{} for t in ('control','mixup')}; fold_results=[]; deltas=[]
    for fold in range(3):
        results, bindings = {}, {}
        for treatment in ('control','mixup'):
            run = ROOT/('runs/v13_fold'+str(fold)+'_'+treatment)
            bindings[treatment] = completed(run,fold,treatment)
            with (run/'validation_predictions.csv').open(encoding='utf-8',newline='') as stream:
                predictions = list(csv.DictReader(stream))
            names = [r['Image_name'] for r in predictions]
            if names != manifest['folds'][fold]['holdout_names']: raise ValueError('OOF order mismatch')
            if any(float(r['True_Label']) != labels[r['Image_name']] for r in predictions): raise ValueError('OOF labels mismatch')
            ps = [float(r['Probability']) for r in predictions]
            results[treatment] = metrics([labels[n] for n in names],ps)
            if set(names)&set(all_values[treatment]): raise ValueError('Repeated OOF row')
            all_values[treatment].update(zip(names,ps))
        if bindings['control']['initial_model_sha256'] != bindings['mixup']['initial_model_sha256']:
            raise ValueError('Unpaired initialization')
        for batch in range(2):
            paths = [ROOT/('runs/v13_fold'+str(fold)+'_'+t)/('batch_identity_'+str(batch)+'.json') for t in ('control','mixup')]
            if paths[0].read_bytes()!=paths[1].read_bytes(): raise ValueError('Unpaired loader streams')
        deltas.append(results['mixup']['auroc']-results['control']['auroc']);fold_results.append(results)
    names = [r['Image_name'] for r in rows]; y = [labels[n] for n in names]
    if any(set(v)!=set(names) for v in all_values.values()): raise ValueError('Incomplete OOF coverage')
    oof = {t:metrics(y,[v[n] for n in names]) for t,v in all_values.items()}
    passed = promotion(deltas,oof['control']['average_precision'],oof['mixup']['average_precision'])
    result = dict(passed=passed,fold_metrics=fold_results,auroc_deltas=deltas,oof_metrics=oof,
                  fixed_epoch=8,rows=len(rows),patient_independence_verified=False,
                  limitations='Exploratory shared-training folds; OOF AP sensitive to score scale; not platform score')
    save_json(OUT/'comparison.json',result)
    with (OUT/'oof_predictions.npz').open('xb') as stream:
        np.savez_compressed(stream,names=np.array(names),labels=np.array(y),**{t:np.array([v[n] for n in names]) for t,v in all_values.items()})
    return result


def finalize():
    frozen(); require_promotion(ROOT); backend(); run = ROOT/'runs/v13_full_mixup'; completed(run,full=True)
    target = OUT/'finalization'; target.mkdir(exist_ok=False)
    config = target/'selected_config.json'; bundle(config,run/'best.pt'); bound = identity(config)
    save_json(target/'frozen_model.json',dict(files=bound))
    timing = measure(config,'v13_final_timing');save_json(target/'local_runtime_check.json',timing)
    probe = json.loads((ROOT/'reports/v13_resource_probe.json').read_text(encoding='utf-8'))
    if not resource_gate(probe['peak_allocated_bytes'],timing['mean_ms'],timing['max_repeat_delta'],timing['runtime_bytes']):
        save_json(target/'result.json',dict(passed=False,reason='Final resource gate failed')); return
    audit_inputs(RAW,ROOT/'data/cache384',384)
    predictor=Predictor(config);baseline=Predictor(ROOT/'output/v6/selected_config.json')
    rows=load_rows(RAW,'valid');paths=image_index(RAW/'valid');names=[];labels=[];values=[];reference=[]
    for i,row in enumerate(rows):
        name=row['Image_name'];names.append(name);labels.append(int(row['True_Label']))
        values.append(predictor.predict_one(paths[name.casefold()]));reference.append(baseline.predict_one(paths[name.casefold()]))
        if i%500==0:print('final validation',i,len(rows),flush=True)
    write_submission(target/'validation_predictions.csv',names,values)
    write_submission(target/'baseline_validation_predictions.csv',names,reference)
    current,old=metrics(labels,values),metrics(labels,reference)
    rng=np.random.default_rng(20261105);y=np.array(labels);a=np.array(values);b=np.array(reference)
    intervals=[];neg=np.flatnonzero(y==0);pos=np.flatnonzero(y==1)
    for _ in range(1000):
        indices=np.concatenate([rng.choice(neg,len(neg)),rng.choice(pos,len(pos))])
        cm,om=metrics(y[indices].tolist(),a[indices].tolist()),metrics(y[indices].tolist(),b[indices].tolist())
        intervals.append([cm['auroc']-om['auroc'],cm['average_precision']-om['average_precision']])
    save_json(target/'validation_report.json',dict(candidate=current,baseline=old,
        paired_bootstrap_percentile=np.percentile(intervals,[2.5,97.5],axis=0).tolist(),seed=20261105,replicates=1000,
        limitations='Descriptive image bootstrap on reused validation; not patient or selection adjusted',platform_score=None))
    del baseline
    frozen();verify_evidence(bound)
    save_json(target/'frozen_trial_inputs.json',dict(files=bound,fixed_epoch=8,selection_on_test=False))
    names=test_names(RAW);paths=image_index(RAW/'test');values=[]
    for i,name in enumerate(names):
        values.append(predictor.predict_one(paths[name.casefold()]))
        if i%500==0:print('test forward',i,len(names),flush=True)
    if len(names)!=5855 or len(set(names))!=5855:raise ValueError('Wrong test names')
    verify_evidence(bound);frozen()
    csv_path=target/'submission.csv';write_submission(csv_path,names,values)
    save_json(target/'submission.manifest.json',dict(runtime_files=bound,submission_sha256=digest(csv_path)))
    destination=ROOT.parent/'第十三版试交_v13'
    destination.mkdir(exist_ok=False)
    for name in ('submission.csv','submission.manifest.json','selected_config.json','validation_report.json','local_runtime_check.json'):
        shutil.copy2(target/name,destination/name)
        if digest(target/name)!=digest(destination/name):raise ValueError('Delivery hash mismatch')
    save_json(target/'result.json',dict(passed=True,delivery=str(destination/'submission.csv'),
        sha256=digest(csv_path),rows=5855,candidate=current,baseline=old,official_T4_verified=False,platform_score=None,
        independent_delivery_review_pending=True))


def run_child(command, label, deadline, maximum=10800):
    timeout=remaining_timeout(deadline,time.time(),maximum)
    with (ROOT/'reports'/('v13_'+label+'.log')).open('x',encoding='utf-8') as stream:
        child=subprocess.Popen([sys.executable,'-u']+command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        try:
            save_json(ROOT/'reports/v13_child.json',dict(pid=child.pid,command=command,started_at=time.time(),deadline=deadline))
            code=child.wait(timeout=timeout)
        except BaseException:
            if child.poll() is None:
                subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],check=True,capture_output=True)
                child.wait(timeout=30)
            raise
    if code:raise RuntimeError('Stage failed: '+label+' exit='+str(code))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('preflight','finalize','accept_delivery'))
    args=parser.parse_args()
    if args.stage=='accept_delivery':
        status=ROOT/'reports/v13_status.json'
        state=json.loads(status.read_text(encoding='utf-8'))
        if state['phase']!='awaiting_review':raise ValueError('No delivery awaiting independent review')
        lock=OUT/'operation.lock'
        with lock.open('x') as stream:stream.write(str(os.getpid()))
        try:
            review=json.loads((ROOT/'reports/v13_delivery_review.json').read_text(encoding='utf-8'))
            if not review['approved']:raise ValueError('Independent delivery review not approved')
            verify_evidence(review['evidence']);frozen()
            result=state['result']
            final=OUT/'finalization'
            bound=json.loads((final/'frozen_trial_inputs.json').read_text(encoding='utf-8'))['files']
            verify_evidence(bound)
            required={str(Path(i['path']).resolve()) for i in bound}
            required.update(str((final/n).resolve()) for n in ('submission.csv','submission.manifest.json',
                            'validation_report.json','local_runtime_check.json','frozen_trial_inputs.json','result.json'))
            required.add(str(Path(result['delivery']).resolve()))
            reviewed={str(Path(i['path']).resolve()) for i in review['evidence']}
            if not required.issubset(reviewed):raise ValueError('Independent review evidence coverage incomplete')
            if review['submission_sha256']!=result['sha256'] or digest(result['delivery'])!=result['sha256']:
                raise ValueError('Review delivery mismatch')
            if digest(final/'submission.csv')!=result['sha256']:raise ValueError('Canonical delivery mismatch')
            save_json(status,transition(state,'complete',review=review))
        finally:
            if lock.exists() and lock.read_text().strip()==str(os.getpid()):lock.unlink()
        return
    if args.stage in ('preflight','finalize'):
        state=json.loads((ROOT/'reports/v13_status.json').read_text(encoding='utf-8'))
        comparison=require_promotion(ROOT) if args.stage=='finalize' else None
        stage_guard(args.stage,state,comparison)
        if (OUT/'operation.lock').read_text().strip()!=str(state['pid']):raise ValueError('Coordinator ownership mismatch')
        lock=OUT/(args.stage+'.lock')
        with lock.open('x') as stream:stream.write(str(os.getpid()))
        try:
            if args.stage=='preflight':
                if (ROOT/'reports/v13_resource_probe.json').exists() or (OUT/'resource_config.json').exists():
                    raise FileExistsError('No repeated resource probe')
                preflight()
            else:finalize()
        finally:
            if lock.exists() and lock.read_text().strip()==str(os.getpid()):lock.unlink()
        return
    status=ROOT/'reports/v13_status.json'
    if OUT.exists() or status.exists():raise FileExistsError('Preserve existing v13 operation')
    OUT.mkdir();lock=OUT/'operation.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    state=dict(phase='implementation',revision=0,history=[],pid=os.getpid(),started_at=time.time())
    deadline=state['started_at']+43200
    def advance(phase,**details):
        nonlocal state
        state=transition(state,phase,**details);save_json(status,state);print(json.dumps(state),flush=True)
    try:
        log=ROOT/'reports/v13_verification.log'
        if 'VERIFY PASSED:' not in log.read_text(encoding='utf-8'):raise ValueError('Verification missing')
        review=json.loads((ROOT/'reports/v13_preflight_review.json').read_text(encoding='utf-8'))
        if not review['approved']:raise ValueError('Independent review incomplete')
        verify_evidence(review['reviewed_sources'])
        paths=list(ROOT.glob('*.py'))+list((ROOT/'tools').glob('*.py'))+list((ROOT/'tests').glob('*.py'))
        paths += [ROOT/n for n in ('docs/ADR-013-mixup-folds.md','docs/science-superpowers/plans/2026-10-29-breakthrough.md',
                 'output/v13_planning/three_fold_manifest.json','data/cache384/READY.json','data/raw/glaucoma_first/train.csv',
                 'data/raw/glaucoma_first/valid.csv','pretrain_model/convnext_tiny-983f1562.pth','reports/v13_verification.log',
                 'reports/v13_preflight_review.json')]
        paths += list(ROOT.parent.glob('第*版*/submission.csv'))
        evidence=[dict(path=str(p),sha256=digest(p)) for p in sorted(set(paths))]+inference_fingerprint(ROOT/'output/v6/selected_config.json')
        save_json(OUT/'frozen_inputs.json',dict(evidence=evidence,deadline=deadline,started_at=state['started_at']))
        advance('preflight',deadline=deadline)
        run_child(['-m','tools.run_v13','--stage','preflight'],'preflight',deadline)
        probe=json.loads((ROOT/'reports/v13_resource_probe.json').read_text(encoding='utf-8'))
        if not probe['passed']:advance('not_promoted',reason='One-shot resource preflight failed');return
        advance('fold_training')
        for fold in range(3):
            for treatment in ('control','mixup'):
                frozen();label='fold'+str(fold)+'_'+treatment
                run_child(['train_mixup.py','--fold',str(fold),'--treatment',treatment,'--run','runs/v13_'+label],label,deadline)
                frozen();completed(ROOT/('runs/v13_'+label),fold,treatment)
        advance('comparison');remaining_timeout(deadline,time.time(),10800);result=compare()
        remaining_timeout(deadline,time.time(),10800)
        if not result['passed']:advance('not_promoted',reason='Three-fold gate failed',comparison=result);return
        promotion_paths=[OUT/'comparison.json',OUT/'oof_predictions.npz']
        for fold in range(3):
            for treatment in ('control','mixup'):
                run=ROOT/('runs/v13_fold'+str(fold)+'_'+treatment)
                promotion_paths += [run/n for n in ('best.pt','config.json','binding.json','status.json','history.jsonl','validation_predictions.csv')]
        save_json(OUT/'promotion.json',dict(comparison_sha256=digest(OUT/'comparison.json'),
            evidence=[dict(path=str(p),sha256=digest(p)) for p in promotion_paths]))
        require_promotion(ROOT)
        deadline=time.time()+14400
        advance('full_training',full_deadline=deadline,comparison=result)
        run_child(['train_mixup.py','--full','--treatment','mixup','--run','runs/v13_full_mixup'],'full_mixup',deadline)
        frozen();completed(ROOT/'runs/v13_full_mixup',full=True)
        advance('finalization')
        run_child(['-m','tools.run_v13','--stage','finalize'],'finalization',deadline,14400)
        result=json.loads((OUT/'finalization/result.json').read_text(encoding='utf-8'));frozen()
        advance('awaiting_review' if result['passed'] else 'not_promoted',result=result)
    except (TimeoutError,subprocess.TimeoutExpired) as error:
        if state['phase'] not in TERMINAL:advance('budget_exceeded',error=repr(error))
        raise
    except BaseException as error:
        if state['phase'] not in TERMINAL:advance('failed',error=repr(error))
        raise
    finally:
        if lock.exists() and lock.read_text().strip()==str(os.getpid()):lock.unlink()


if __name__=='__main__':main()
