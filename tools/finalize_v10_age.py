"""Finite v10 full-model validation, resource audit and one fixed trial CSV."""
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from clinical_predict import Predictor,fingerprint,write_age_metadata
from data import audit_inputs,load_rows,image_index
from predict import Predictor as ImagePredictor,test_names
from provenance import digest,inference_fingerprint
from utils import metrics,save_json,write_submission
from v8_integrity import verify_evidence
from v10_integrity import frozen,recipe,validate_completed_age
from tools.bootstrap_comparison import main as bootstrap


def trial_allowed(phase,mean_ms,runtime_bytes):
    return (phase=='full_ready' and math.isfinite(mean_ms) and 0<mean_ms<=100 and
            type(runtime_bytes) is int and 0<runtime_bytes<=500_000_000)


def stage_delivery(source,destination,mappings,report,note):
    source,destination=Path(source),Path(destination)
    if destination.exists():raise FileExistsError('Preserve existing trial delivery')
    staging=source/'delivery_staging';staging.mkdir(exist_ok=False)
    for original,name in mappings.items():
        shutil.copy2(source/original,staging/name)
        if digest(source/original)!=digest(staging/name):raise ValueError('Clinical delivery copy mismatch')
    save_json(staging/'交付校验.json',report)
    (staging/'上传说明.txt').write_text(note,encoding='utf-8')
    # All copies and sidecars exist before the official delivery directory appears.
    # Windows rejects replacement if another writer has created the destination.
    staging.rename(destination)


def main():
    output=ROOT/'output/v10';status=ROOT/'reports/v10_finalization_status.json'
    if status.exists() or (output/'finalization').exists():raise FileExistsError('Preserve v10 finalization')
    target=output/'finalization';target.mkdir()
    current='ready';history=[]
    allowed={'ready':{'waiting_for_full','failed'},'waiting_for_full':{'timing','not_promoted','failed'},
        'timing':{'validating','not_promoted','failed'},'validating':{'predicting','failed'},
        'predicting':{'checking','failed'},'checking':{'complete','failed'},'complete':set(),'not_promoted':set(),'failed':set()}
    def advance(phase,**details):
        nonlocal current
        if phase not in allowed[current]:raise ValueError('Illegal v10 finalization transition')
        history.append(dict(phase=phase,time=time.time(),details=details));current=phase
        save_json(status,dict(phase=phase,history=history,**details))
    try:
        frozen(ROOT)
        files=[ROOT/name for name in ('tools/finalize_v10_age.py','tools/bootstrap_comparison.py',
            'tests/test_age_finalization.py','tests/test_age_mixed_bundle.py',
            'docs/FINALIZATION-v10.md','reports/v10_finalizer_verification.log',
            'output/v10/frozen_inputs.json','data/raw/glaucoma_first/valid.csv')]
        evidence=[dict(path=str(path),sha256=digest(path)) for path in files]
        baseline_config=ROOT/'output/v6/selected_config.json';evidence+=inference_fingerprint(baseline_config)
        save_json(target/'frozen_stage_inputs.json',dict(evidence=evidence,source='fixed plan before full-ready validation',
            policy='One fixed full candidate; validation uncertainty/regression disclosed, no test-driven choice'))
        advance('waiting_for_full');deadline=time.monotonic()+43200
        while True:
            operation=json.loads((ROOT/'reports/v10_status.json').read_text(encoding='utf-8'))
            phase=operation['phase']
            if phase=='not_promoted':advance('not_promoted',reason=operation.get('reason'),gate=operation.get('gate'));return
            if phase=='failed':raise RuntimeError('V10 training/comparison failed: '+operation.get('error',''))
            if phase=='full_ready':break
            if time.monotonic()>deadline:raise TimeoutError('V10 finite finalization wait exceeded12hours')
            time.sleep(10)
        verify_evidence(evidence);frozen(ROOT)
        selection=json.loads((output/'internal_selection.json').read_text(encoding='utf-8'))
        verify_evidence(selection['evidence'])
        if not selection['gate']['passed']:raise ValueError('Internal Age gate is not qualified')
        checked=validate_completed_age(ROOT/'runs/v10_full_age',recipe('runs/v10_full_age',True,selection['selected_epoch']),ROOT)
        raw=ROOT/'data/raw/glaucoma_first';audit=audit_inputs(raw,ROOT/'data/cache384',384)
        for name in ('best.pt','config.json','age_stats.json','best_metrics.json','status.json','split_binding.json'):
            path=ROOT/'runs/v10_full_age'/name;evidence.append(dict(path=str(path),sha256=digest(path)))
        age_model=ROOT/'runs/v10_full_age/best.pt';r50=ROOT/'runs/v3_r50_384_w1/best.pt'
        metadata=target/'valid_age.csv';write_age_metadata(raw/'valid.csv',metadata)
        missing=target/'missing_age.csv';missing.write_text('Image_name,Age\n',encoding='utf-8')
        members=[dict(checkpoint=str(r50),sha256=digest(r50),tta=True),dict(checkpoint=str(age_model),sha256=digest(age_model),tta=True)]
        def config(path,age_table):
            save_json(path,dict(schema_version=2,name='v10_fixed_equal_r50_age_convnext_tta',models=members,
                age_metadata=age_table.name,age_metadata_sha256=digest(age_table),batch=1,precision='fp16',
                probability_precision='fp32',threshold=.5,selection_source='unique plan-frozen candidate after train-holdout qualification'))
        valid_config=target/'valid_config.json';config(valid_config,metadata)
        missing_config=target/'missing_config.json';config(missing_config,missing)
        evidence+=fingerprint(valid_config)+fingerprint(missing_config)
        torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        rows=load_rows(raw,'valid');paths=image_index(raw/'valid')
        predictor=Predictor(valid_config)
        advance('timing')
        for row in rows[:5]:predictor.predict_one(paths[row['Image_name'].casefold()])
        directory=ROOT/'tmp/v10_final_timing';directory.mkdir(exist_ok=True)
        rounds=[];prior={};delta=0.
        for repeat in range(3):
            durations=[]
            for row in rows[5:105]:
                name=row['Image_name'];torch.cuda.synchronize();start=time.perf_counter()
                p=predictor.predict_one(paths[name.casefold()]);write_submission(directory/(name+'.csv'),[name],[p])
                torch.cuda.synchronize();durations.append((time.perf_counter()-start)*1000)
                if name in prior:delta=max(delta,abs(p-prior[name]))
                prior[name]=p
            rounds.append(durations)
        mean=sum(map(sum,rounds))/300
        runtime_bytes=sum(path.stat().st_size for path in (r50,age_model,metadata,valid_config))
        runtime_bytes+=sum((ROOT/name).stat().st_size for name in ('clinical.py','clinical_predict.py','data.py','model.py','utils.py','provenance.py','predict.py','cost.py'))
        timing=dict(mean_ms=mean,limit_ms=100,rounds_ms=rounds,max_repeat_delta=delta,batch=1,warmup=5,samples=100,
                    rounds=3,runtime_bytes=runtime_bytes,official_T4_verified=False)
        save_json(target/'local_runtime_check.json',timing)
        if not trial_allowed(phase,mean,runtime_bytes) or delta>1e-7:
            advance('not_promoted',reason='Final fixed-model resource gate failed',timing=timing);return
        advance('validating')
        baseline=ImagePredictor(baseline_config);fallback=Predictor(missing_config)
        names=[];labels=[];probabilities=[];reference=[];without_age=[]
        for index,row in enumerate(rows,1):
            name=row['Image_name'];path=paths[name.casefold()]
            names.append(name);labels.append(int(row['True_Label']))
            probabilities.append(predictor.predict_one(path));reference.append(baseline.predict_one(path));without_age.append(fallback.predict_one(path))
            if index%500==0:print('validated',index,'/',len(rows),flush=True)
        write_submission(target/'validation_predictions.csv',names,probabilities)
        write_submission(target/'baseline_single_image_validation.csv',names,reference)
        write_submission(target/'missing_age_validation.csv',names,without_age)
        point=metrics(labels,probabilities);old=metrics(labels,reference);missing_metrics=metrics(labels,without_age)
        save_json(target/'selection_report.json',dict(name='v10_fixed_equal_r50_age_convnext_tta',reference_name='fixed_v6',
            actual_single_image_validation=point,baseline_single_image_validation=old,missing_age_validation=missing_metrics,
            auroc_gain=point['auroc']-old['auroc'],average_precision_gain=point['average_precision']-old['average_precision'],
            model_selection_performed=False,trial_csv_policy='Fixed candidate saved after resources regardless of point-score direction',
            official_T4_verified=False,limitations='Repeated internal/officialvalid use, image-level dependence and Age/source confounding; no platform guarantee'))
        bootstrap(target,labels_path=raw/'valid.csv')
        if audit_inputs(raw,ROOT/'data/cache384',384)!=audit:raise ValueError('Image input audit changed during full validation')
        verify_evidence(evidence);frozen(ROOT)
        test_metadata=target/'test_age.csv';write_age_metadata(raw/'test.csv',test_metadata)
        trial_config=target/'trial_config.json';config(trial_config,test_metadata)
        trial_identity=fingerprint(trial_config);save_json(target/'frozen_trial_inputs.json',dict(files=trial_identity,
            image_models_and_weights_frozen_before_test=True,age_stats_source='officialtrain only; no test statistics'))
        advance('predicting')
        with (ROOT/'reports/v10_test_prediction.log').open('x',encoding='utf-8') as stream:
            subprocess.run([sys.executable,'-u','clinical_predict.py','--checkpoint',str(trial_config),
                            '--output',str(target/'submission.csv')],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        advance('checking');verify_evidence(trial_identity);verify_evidence(evidence)
        submission=target/'submission.csv';manifest=json.loads(submission.with_suffix('.manifest.json').read_text(encoding='utf-8'))
        if manifest['files']!=trial_identity or manifest['submission_sha256']!=digest(submission):raise ValueError('Clinical prediction binding mismatch')
        expected=test_names(raw)
        with submission.open(encoding='utf-8',newline='') as stream:
            reader=csv.DictReader(stream)
            if reader.fieldnames!=['Image_name','Probability','Pred_Label']:raise ValueError('Invalid clinical CSV header')
            predicted=list(reader)
        if [row['Image_name'] for row in predicted]!=expected or len(predicted)!=5855:raise ValueError('Clinical CSV names/count mismatch')
        for row in predicted:
            p=float(row['Probability'])
            if not math.isfinite(p) or not 0<=p<=1 or row['Pred_Label']!=str(int(p>=.5)):raise ValueError('Clinical CSV probability/label mismatch')
        destination=ROOT.parent/'第十版试交_v10'
        mappings={'submission.csv':'submission.csv','trial_config.json':'模型配置.json','test_age.csv':'test_age.csv',
            'submission.manifest.json':'模型输出绑定.json','selection_report.json':'验证结果.json',
            'bootstrap_comparison.json':'验证不确定性.json','local_runtime_check.json':'本机测速结果.json'}
        report=dict(rows=5855,csv_sha256=digest(submission),files=trial_identity,runtime_bytes=runtime_bytes,
            model_and_metadata_binding_verified=True,full_retraining_performed=True,
            actual_single_image_validation=point,baseline=old,local_mean_ms=mean,official_T4_verified=False,platform_score=None)
        note='第十版：固定R50_TTA与图像加年龄ConvNeXt_TTA等权融合。\n评分入口上传submission.csv，保持文件名。\n年龄统计只由官方train拟合，测试Age仅作为该图片的输入；缺失年龄按固定规则处理。\n已按内部选定轮次完成全量重训。验证结果和描述性不确定性见附带报告；不能保证线上上涨或达到99.3。\n本机速度符合本次记录的100ms检查，官方T4/指定环境尚未验证。\n此目录不是核验ZIP；后续材料必须匹配本次CSV、模型和Age元数据。旧版文件保留。\n'
        stage_delivery(target,destination,mappings,report,note)
        advance('complete',delivery=str(destination/'submission.csv'),verification=report)
        print('V10 TRIAL CSV READY',str(destination/'submission.csv'),flush=True)
    except BaseException as error:
        if current not in ('complete','not_promoted','failed'):advance('failed',error=repr(error))
        raise


if __name__=='__main__':main()
