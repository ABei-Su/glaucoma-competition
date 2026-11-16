"""Finite fixed384-pair resource/validation audit and one trial delivery."""
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data import audit_inputs,load_rows,image_index
from predict import Predictor,test_names
from provenance import digest,inference_fingerprint
from utils import metrics,save_json,write_submission
from tools.bootstrap_comparison import main as bootstrap
from tools.finalize_v10_age import stage_delivery
from tools.predict_v12 import configure_backend
from tools.run_v12 import bundle,measure
from v12_protocol import transition,resource_gate,verify_evidence,frozen,validate_completed,inference_identity


def delivery_directory(root=ROOT):
    return Path(root).parent/'第十二版试交_v12'


def main():
    output=ROOT/'output/v12';status=ROOT/'reports/v12_finalization_status.json';target=output/'finalization'
    if status.exists() or target.exists():raise FileExistsError('Preserve existing v12 finalization')
    # Coordinator creates output first; its normal import/preflight may take time.
    deadline=time.monotonic()+43200
    while not output.exists():
        if time.monotonic()>deadline:raise TimeoutError('No v12 coordinator output within12hours')
        time.sleep(5)
    lock=output/'finalization.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    state=dict(kind='finalization',phase='ready',revision=0,history=[])
    def advance(phase,**details):
        nonlocal state
        state=transition(state,phase,details);save_json(status,state);print('V12 FINALIZATION',phase,flush=True)
    try:
        target.mkdir(exist_ok=False)
        sources=list(ROOT.glob('*.py'))+list((ROOT/'tools').glob('*.py'))+list((ROOT/'tests').glob('*.py'))
        sources += [ROOT/name for name in ('docs/OPTIMIZATION-v12.md','docs/ADR-012-focal-training.md',
            'docs/REVIEW-v12-preflight.md','reports/v12_verification.log','data/raw/glaucoma_first/valid.csv')]
        evidence=[dict(path=str(path),sha256=digest(path)) for path in sorted(set(sources))]
        baseline_config=ROOT/'output/v6/selected_config.json';evidence+=inference_fingerprint(baseline_config)
        save_json(target/'frozen_stage_inputs.json',dict(evidence=evidence,policy='Unique fixed full candidate, no post-validation choice'))
        advance('waiting_for_full')
        while True:
            operation_path=ROOT/'reports/v12_status.json'
            if operation_path.exists():
                operation=json.loads(operation_path.read_text(encoding='utf-8'))
                if operation['phase']=='not_promoted':
                    advance('not_promoted',reason=operation.get('reason'),gate=operation.get('gate'));return
                if operation['phase']=='failed':raise RuntimeError('V12 experiment failed: '+operation.get('error',''))
                if operation['phase']=='full_ready':break
            if time.monotonic()>deadline:raise TimeoutError('Finite v12 finalizer wait exceeded12hours')
            time.sleep(10)
        verify_evidence(evidence);record=frozen(ROOT)
        selection=json.loads((output/'internal_selection.json').read_text(encoding='utf-8'))
        verify_evidence(selection['evidence'])
        if not selection['gate']['passed'] or operation['frozen_epoch']!=selection['selected_epoch']:
            raise ValueError('Full epoch/qualification changed')
        epoch=selection['selected_epoch'];validate_completed(ROOT,False,epoch)
        for name in ('best.pt','config.json','best_metrics.json','status.json','history.jsonl','environment.json'):
            path=ROOT/'runs/v12_full_focal'/name;evidence.append(dict(path=str(path),sha256=digest(path)))
        raw=ROOT/'data/raw/glaucoma_first';audit=audit_inputs(raw,ROOT/'data/cache384',384)
        config=target/'selected_config.json';identity=bundle(config,ROOT/'runs/v12_full_focal/best.pt')
        evidence+=identity;save_json(target/'frozen_full_model.json',dict(evidence=evidence,selected_epoch=epoch))
        advance('timing',full_ready=True)
        timing=measure(config,'v12_final_timing');save_json(target/'local_runtime_check.json',timing)
        resource=json.loads((ROOT/'reports/v12_resource_probe.json').read_text(encoding='utf-8'))
        if not resource_gate(resource['capacity']['peak_allocated_bytes'],timing['mean_ms'],timing['max_repeat_delta'],timing['runtime_bytes']):
            advance('not_promoted',reason='Fixed final pair resource gate failed',timing=timing);return
        verify_evidence(evidence);advance('validating');configure_backend()
        predictor=Predictor(config);baseline=Predictor(baseline_config)
        rows=load_rows(raw,'valid');paths=image_index(raw/'valid');names=[];labels=[];values=[];reference=[]
        for index,row in enumerate(rows,1):
            name=row['Image_name'];path=paths[name.casefold()];names.append(name);labels.append(int(row['True_Label']))
            values.append(predictor.predict_one(path));reference.append(baseline.predict_one(path))
            if index%500==0:print('validated',index,'/',len(rows),flush=True)
        write_submission(target/'validation_predictions.csv',names,values)
        write_submission(target/'baseline_single_image_validation.csv',names,reference)
        current=metrics(labels,values);old=metrics(labels,reference)
        save_json(target/'selection_report.json',dict(name='v12_fixed_equal_r50_384_cn384_focal_tta',reference_name='fixed_v6',
            actual_single_image_validation=current,baseline_single_image_validation=old,auroc_gain=current['auroc']-old['auroc'],
            average_precision_gain=current['average_precision']-old['average_precision'],model_selection_performed=False,
            trial_csv_policy='Unique fixed full candidate saved after resource checks; any regression disclosed',
            official_T4_verified=False,limitations='Reused internal/officialvalid; unavailable patient groups; no leaderboard/clinical claim'))
        bootstrap(target,labels_path=raw/'valid.csv')
        if audit_inputs(raw,ROOT/'data/cache384',384)!=audit:raise ValueError('Validation source/cache content changed')
        verify_evidence(evidence);frozen(ROOT)
        if identity!=inference_identity(config):raise ValueError('Fixed config changed before test')
        save_json(target/'frozen_trial_inputs.json',dict(files=identity,selected_epoch=epoch,
            model_selection_performed=False,source='One fixed trained pair; no test/leaderboard adaptation'))
        advance('predicting',resource_passed=True,trial_frozen=True)
        with (ROOT/'reports/v12_test_prediction.log').open('x',encoding='utf-8') as stream:
            subprocess.run([sys.executable,'-u','-m','tools.predict_v12','--checkpoint',str(config),
                            '--output',str(target/'submission.csv')],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        advance('checking');verify_evidence(evidence);frozen(ROOT)
        if identity!=inference_identity(config):raise ValueError('Prediction identity changed')
        submission=target/'submission.csv';manifest=json.loads(submission.with_suffix('.manifest.json').read_text(encoding='utf-8'))
        if manifest['runtime_files']!=identity or manifest['submission_sha256']!=digest(submission):raise ValueError('Prediction binding failed')
        with submission.open(encoding='utf-8',newline='') as stream:
            reader=csv.DictReader(stream)
            if reader.fieldnames!=['Image_name','Probability','Pred_Label']:raise ValueError('Wrong submission header')
            predictions=list(reader)
        if len(predictions)!=5855 or [r['Image_name'] for r in predictions]!=test_names(raw):raise ValueError('Wrong submission names/count')
        for row in predictions:
            p=float(row['Probability'])
            if not math.isfinite(p) or not 0<=p<=1 or row['Pred_Label']!=str(int(p>=.5)):raise ValueError('Invalid probability/threshold label')
        destination=delivery_directory(ROOT)
        report=dict(rows=5855,csv_sha256=digest(submission),files=identity,full_retraining_performed=True,
            frozen_epoch=epoch,runtime_bytes=timing['runtime_bytes'],local_mean_ms=timing['mean_ms'],
            actual_single_image_validation=current,baseline=old,official_T4_verified=False,platform_score=None)
        mappings={'submission.csv':'submission.csv','selected_config.json':'模型配置.json','submission.manifest.json':'模型输出绑定.json',
                  'selection_report.json':'验证结果.json','bootstrap_comparison.json':'验证不确定性.json','local_runtime_check.json':'本机测速结果.json'}
        note=('第十二版试交：固定R50_384_TTA与ConvNeXt384_Focal_TTA等权概率融合。\n'
              '评分入口只上传submission.csv，保持文件名；用户自行上传。\n'
              '仅主办方train使用固定Focal Loss训练，按内部选定轮次全量重训；无年龄或测试统计拟合。\n'
              '完整验证结果和描述性区间见附带报告，可能退步；不保证线上超过98.60或达到99.3。\n'
              '本机测速通过，官方T4/指定环境未验证。此目录不是模型核验ZIP；旧版全部保留。\n')
        stage_delivery(target,destination,mappings,report,note)
        advance('complete',rows=5855,delivery_verified=True,delivery=str(destination/'submission.csv'),verification=report)
    except BaseException as error:
        if state['phase'] not in ('failed','not_promoted','complete'):advance('failed',error=repr(error))
        raise
    finally:
        if lock.exists() and lock.read_text().strip()==str(os.getpid()):lock.unlink()


if __name__=='__main__':main()
