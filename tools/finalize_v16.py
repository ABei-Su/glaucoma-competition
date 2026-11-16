"""One-shot, bounded finalization of the predeclared v16 single and pair trials."""
import csv
import json
import math
import os
import shutil
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/v16/finalization'
RAW=ROOT/'data/raw/glaucoma_first'
RUN=ROOT/'runs/v16_r50_refine4'
SOURCES=['predict.py','data.py','model.py','utils.py','provenance.py','cache_paths.py',
         'tools/finalize_v16.py','tests/test_v16_finalization.py']


def check_training(config,status,checkpoint,epochs):
    expected=dict(architecture='resnet50',cache='data/cache384',run='runs/v16_r50_refine4',
        epochs=4,checkpoint_epoch=4,size=384,batch=16,workers=4,seed=20261033,lr=.00001,
        pos_weight=1.,augmentation='mild',memory_format='contiguous',
        validation_probability_precision='fp32',split_manifest=None,init_checkpoint='runs/v15_full_r50_8/best.pt',
        data='data/raw/glaucoma_first',pretrain='pretrain_model/resnet50-11ad3fa6.pth',
        max_steps=0,resume=False)
    if any(config.get(k)!=v for k,v in expected.items()):raise ValueError('Training recipe changed')
    if status.get('state')!='complete' or status['best']['epoch']!=4 or epochs!=list(range(1,5)):
        raise ValueError('Training incomplete')
    if (checkpoint.get('checkpoint_epoch')!=4 or checkpoint['metrics']['epoch']!=4 or
        checkpoint.get('architecture')!='resnet50' or checkpoint.get('size')!=384 or
        checkpoint.get('validation_partition')!='official_valid'):
        raise ValueError('Checkpoint identity mismatch')


def check_ancestry(binding,expected_sha,initial,final,data_sha):
    if (binding.get('checkpoint')!='runs/v15_full_r50_8/best.pt' or binding.get('sha256')!=expected_sha or
        binding.get('kind')!='supervised continuation on official train only'):
        raise ValueError('Supervised initialization identity mismatch')
    if (initial.get('architecture')!='resnet50' or initial['metrics']['epoch']!=8 or
        initial.get('checkpoint_epoch')!=8 or initial.get('validation_partition')!='official_valid' or
        initial.get('data_fingerprint')!=data_sha or final.get('data_fingerprint')!=data_sha):
        raise ValueError('Ancestry stage or data mismatch')


def resource_passes(mean_ms,delta,size):
    return all(math.isfinite(x) for x in (mean_ms,delta,size)) and 0<=mean_ms<=100 and 0<=delta<=1e-7 and 0<size<=500000000


def repeat_delta(values):
    if not values or not all(math.isfinite(v) for v in values):return float('inf')
    return max(values)-min(values)


def next_phase(current,destination):
    edges={'created':'resources','resources':'validation','validation':'prediction','prediction':'awaiting_review'}
    if current not in edges or (destination!='failed' and edges[current]!=destination):
        raise ValueError('Illegal finalization transition')
    return destination


def main():
    import torch
    from data import image_index,load_rows
    from predict import Predictor,test_names
    from provenance import digest,inference_fingerprint
    from utils import save_json,write_submission,metrics
    def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    def verify(items):
        for item in items:
            if digest(item['path'])!=item['sha256']:raise ValueError('Changed input: '+item['path'])
    preflight=read(ROOT/'reports/v16_finalizer_review.json')
    if not preflight['approved']:raise ValueError('Review missing')
    verify(preflight['files'])
    frozen=read(ROOT/'output/v16/frozen_inputs.json');verify(frozen['files'])
    if (RUN/'training.lock').exists():raise ValueError('Training lock still present')
    if read(ROOT/'reports/v16_exit.json').get('exit_code')!=0:raise ValueError('Training did not exit successfully')
    config=read(RUN/'config.json');status=read(RUN/'status.json')
    saved=torch.load(RUN/'best.pt',map_location='cpu',weights_only=True)
    history=[json.loads(line) for line in (RUN/'history.jsonl').read_text().splitlines()]
    check_training(config,status,saved,[r['epoch'] for r in history])
    if saved['metrics']!=status['best'] or saved['metrics']!=history[-1]:raise ValueError('Metrics binding mismatch')
    if not all(torch.isfinite(v).all().item() for v in saved['model'].values() if v.is_floating_point()):
        raise ValueError('Nonfinite checkpoint')
    initial=torch.load(ROOT/'runs/v15_full_r50_8/best.pt',map_location='cpu',weights_only=True)
    check_ancestry(read(RUN/'initialization.json'),digest(ROOT/'runs/v15_full_r50_8/best.pt'),initial,saved,read(ROOT/'data/cache384/READY.json')['manifest_sha256'])
    del initial
    del saved
    OUT.mkdir(exist_ok=False)
    started=time.time();deadline=started+14400;phase='created';events=[]
    state_path=ROOT/'reports/v16_finalization_status.json'
    def persist(**extra):
        save_json(state_path,dict(phase=phase,pid=os.getpid(),started_at=started,deadline=deadline,history=events,**extra))
    def advance(destination,**extra):
        nonlocal phase
        phase=next_phase(phase,destination);events.append(dict(phase=phase,time=time.time()));persist(**extra)
    def check_deadline():
        if time.time()>=deadline:raise TimeoutError('Four-hour finalization budget exceeded')
    persist()
    bound=[]
    try:
        torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        configs={}
        members={'single':[RUN/'best.pt'],'pair':[RUN/'best.pt',ROOT/'runs/v5_full/best.pt']}
        for name,paths in members.items():
            p=OUT/(name+'_config.json')
            save_json(p,dict(schema_version=1,name='v16_'+name,models=[dict(checkpoint=str(x),sha256=digest(x),tta=True) for x in paths],
                batch=1,precision='fp16',probability_precision='fp32',threshold=.5,preprocess_version=1))
            configs[name]=p
        configs['baseline_single']=ROOT/'output/v15/finalization/single_config.json'
        configs['baseline_pair']=ROOT/'output/v15/finalization/pair_config.json'
        files={str(ROOT/n) for n in SOURCES}
        for p in configs.values():files.update(i['path'] for i in inference_fingerprint(p))
        files.update(str(RUN/n) for n in ('config.json','status.json','history.jsonl','environment.json','initialization.json'))
        files.update(str(RAW/n) for n in ('train.csv','valid.csv','test.csv'))
        # Bind actual input bytes, not just CSV names. Hashing is provenance only;
        # test pixels never inform selection or fitting.
        for split in ('valid','test'):
            index=image_index(RAW/split)
            with (RAW/(split+'.csv')).open(encoding='utf-8-sig',newline='') as stream:
                input_names=[row['Image_name'] for row in csv.DictReader(stream)]
            if len(input_names)!=len(set(input_names)):raise ValueError('Duplicate input name')
            files.update(str(index[name.casefold()].resolve()) for name in input_names)
        files.update(str(p.resolve()) for p in ROOT.parent.glob('第*版*/submission.csv'))
        files.add(str(ROOT/'output/v16/frozen_inputs.json'));files.add(str(ROOT/'reports/v16_finalizer_review.json'))
        bound=[dict(path=p,sha256=digest(p)) for p in sorted(files)]
        save_json(OUT/'frozen_trial_inputs.json',dict(files=bound,epoch=4,training_scope='official train only',
            training_rows=51396,validation_rows=5849,selection_on_test=False))
        rows=load_rows(RAW,'valid');paths=image_index(RAW/'valid')
        if len(rows)!=5849 or len(load_rows(RAW,'train'))!=51396:raise ValueError('Unexpected split sizes')
        advance('resources');resources={}
        for kind in ('single','pair'):
            check_deadline();verify(bound);predictor=Predictor(configs[kind])
            for row in rows[100:105]:predictor.predict_one(paths[row['Image_name'].casefold()])
            directory=ROOT/'tmp'/('v16_final_timing_'+kind);directory.mkdir(exist_ok=False)
            rounds=[];observations={};delta=0.
            for repeat in range(3):
                durations=[]
                for row in rows[:100]:
                    check_deadline();name=row['Image_name'];torch.cuda.synchronize();t=time.perf_counter()
                    value=predictor.predict_one(paths[name.casefold()]);write_submission(directory/(name+'.csv'),[name],[value])
                    torch.cuda.synchronize();durations.append((time.perf_counter()-t)*1000)
                    observations.setdefault(name,[]).append(value)
                    delta=max(delta,repeat_delta(observations[name]))
                rounds.append(durations)
            runtime={Path(m['checkpoint']).resolve() for m in read(configs[kind])['models']};runtime.add(configs[kind].resolve())
            size=sum(p.stat().st_size for p in runtime);mean=sum(map(sum,rounds))/300
            resources[kind]=dict(mean_ms=mean,max_repeat_delta=delta,runtime_bytes=size,rounds_ms=rounds,
                passed=resource_passes(mean,delta,size),gpu=torch.cuda.get_device_name(0),official_T4_verified=False,
                timed_names=[r['Image_name'] for r in rows[:100]],warmup_names=[r['Image_name'] for r in rows[100:105]],
                protocol='batch1 raw image to CSV including sync; 300-observation arithmetic mean')
            save_json(OUT/(kind+'_runtime.json'),resources[kind]);del predictor;torch.cuda.empty_cache();verify(bound)
        if not all(r['passed'] for r in resources.values()):
            raise ValueError('Fixed resource gate failed; no automatic remeasurement')
        advance('validation');results={};names=[r['Image_name'] for r in rows];y=[int(r['True_Label']) for r in rows]
        for kind in ('single','pair','baseline_single','baseline_pair'):
            verify(bound);predictor=Predictor(configs[kind]);values=[]
            for i,name in enumerate(names,1):
                check_deadline();values.append(predictor.predict_one(paths[name.casefold()]))
                if i%500==0:print('validation',kind,i,flush=True)
            write_submission(OUT/(kind+'_validation.csv'),names,values);results[kind]=metrics(y,values)
            del predictor;torch.cuda.empty_cache();verify(bound)
        improvements={k:results[k]['auroc']>results['baseline_'+k]['auroc'] and
            results[k]['average_precision']>=results['baseline_'+k]['average_precision'] for k in ('single','pair')}
        save_json(OUT/'validation_report.json',dict(metrics=results,joint_local_improvement=improvements,
            platform_score=None,limitations='Reused validation; exploratory, not independent confirmation'))
        advance('prediction');verify(bound);names=test_names(RAW);paths=image_index(RAW/'test')
        for kind in ('single','pair'):
            predictor=Predictor(configs[kind]);values=[]
            for i,name in enumerate(names,1):
                check_deadline();values.append(predictor.predict_one(paths[name.casefold()]))
                if i%500==0:print('test',kind,i,flush=True)
            verify(bound);verify(frozen['files']);target=OUT/kind;target.mkdir(exist_ok=False)
            write_submission(target/'submission.csv',names,values)
            save_json(target/'submission.manifest.json',dict(files=bound,submission_sha256=digest(target/'submission.csv'),
                runtime_report=str(OUT/(kind+'_runtime.json')),config=str(configs[kind]),kind=kind))
            del predictor;torch.cuda.empty_cache()
        check_deadline();verify(bound);verify(frozen['files'])
        destination=ROOT.parent/'第十六版试交_v16';destination.mkdir(exist_ok=False)
        deliveries={}
        for kind in ('single','pair'):
            folder=destination/kind;folder.mkdir()
            for name in ('submission.csv','submission.manifest.json'):
                shutil.copy2(OUT/kind/name,folder/name)
                if digest(OUT/kind/name)!=digest(folder/name):raise ValueError('Copy mismatch')
            deliveries[kind]=dict(path=str(folder/'submission.csv'),sha256=digest(folder/'submission.csv'))
        save_json(OUT/'result.json',dict(deliveries=deliveries,independent_review_pending=True,
            resources_passed=True,platform_score=None,official_T4_verified=False))
        advance('awaiting_review',deliveries=deliveries)
    except BaseException as exc:
        if phase!='awaiting_review':advance('failed',error=repr(exc))
        raise


if __name__=='__main__':main()
