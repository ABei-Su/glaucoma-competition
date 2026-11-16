"""Package the frozen optimized selection, preserving baseline materials."""
import csv
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from predict import test_names
from utils import save_json
from provenance import verify_prediction_binding


def validate_rows(rows,names):
    if [r['Image_name'] for r in rows]!=names or len(set(names))!=len(names):
        raise ValueError('Submission names/order mismatch')
    for row in rows:
        p=float(row['Probability'])
        if not 0<=p<=1 or row['Pred_Label']!=str(int(p>=.5)):
            raise ValueError('Invalid probability or label')


def main(output=None,provenance_path=None,verification_log=None,training_description=None,extra_initialization=None,
         extra_reports=None,verify_runtime=False,include_baseline_initialization=True,validation_description=None,
         initialization_description=None):
    output=Path(output) if output is not None else ROOT/'output/optimized'
    provenance_path=Path(provenance_path) if provenance_path else ROOT/'reports/pretrain_provenance.json'
    binding=verify_prediction_binding(output/'submission.csv',output/'selected_config.json')
    selection=json.loads((output/'selection_report.json').read_text(encoding='utf-8'))
    if selection['auroc_gain_over_baseline']<0:
        raise ValueError('Selection has not passed validation')
    with (output/'submission.csv').open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=['Image_name','Probability','Pred_Label']:
            raise ValueError('Submission columns mismatch')
        rows=list(reader)
    validate_rows(rows,test_names(ROOT/'data/raw/glaucoma_first'))
    package=output/'materials'
    if package.exists():
        raise FileExistsError('Preserve existing material directory')
    for folder in ['model','code','reports','licenses','pretrain_model','training_initialization']:
        (package/folder).mkdir(parents=True,exist_ok=True)
    if include_baseline_initialization:
        shutil.copy2(ROOT/'runs/baseline_r18_224/best.pt',package/'training_initialization/baseline_best.pt')
    baseline_reports=package/'reports/baseline_initialization'
    baseline_reports.mkdir()
    for name in ['config.json','environment.json','best_metrics.json','history.jsonl']:
        shutil.copy2(ROOT/'runs/baseline_r18_224'/name,baseline_reports/name)
    for source,filename in (extra_initialization or []):
        shutil.copy2(source,package/'training_initialization'/filename)
    for file in ROOT.glob('*.py'):
        shutil.copy2(file,package/'code'/file.name)
    for file in ROOT.glob('requirements*.txt'):
        shutil.copy2(file,package/'code'/file.name)
    config=json.loads((output/'selected_config.json').read_text(encoding='utf-8'))
    for index,member in enumerate(config['models']):
        source=(output/member['checkpoint']).resolve()
        target=f'member_{index}.pt'
        shutil.copy2(source,package/'model'/target)
        member['checkpoint']=target
        report_dir=package/'reports'/source.parent.name
        report_dir.mkdir()
        for name in ['config.json','environment.json','best_metrics.json','history.jsonl','initialization.json',
                     'split_binding.json','resume_events.jsonl']:
            if (source.parent/name).exists():
                shutil.copy2(source.parent/name,report_dir/name)
    for source,_ in (extra_initialization or []):
        report_dir=package/'reports'/source.parent.name
        report_dir.mkdir(exist_ok=True)
        for name in ['config.json','environment.json','best_metrics.json','history.jsonl','initialization.json','resume_events.jsonl']:
            if (source.parent/name).exists(): shutil.copy2(source.parent/name,report_dir/name)
    for report in (extra_reports or []):
        shutil.copy2(report,package/'reports'/report.name)
    save_json(package/'model/config.json',config)
    # Preserve the organizer's three-positional-argument invocation.
    (package/'cost.py').write_text('import runpy,sys\nfrom pathlib import Path\nroot=Path(__file__).resolve().parent\nsys.path.insert(0,str(root/"code"))\nif "--checkpoint" not in sys.argv: sys.argv.extend(["--checkpoint",str(root/"model/config.json")])\nrunpy.run_module("cost",run_name="__main__")\n',encoding='utf-8')
    for name in ['submission.csv','模型核验材料与数据申明清单.xlsx']:
        shutil.copy2(output/name,package/name)
    save_json(package/'reports/prediction_binding.json',binding)
    for name in ['candidate_comparison.json','selection_report.json','local_timing_candidates.json','bootstrap_comparison.json']:
        shutil.copy2(output/name,package/'reports'/name)
    provenance=json.loads(provenance_path.read_text(encoding='utf-8'))
    for item in provenance:
        source=ROOT/item['path']
        if hashlib.sha256(source.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Pretraining provenance mismatch')
        shutil.copy2(source,package/item['path'])
    shutil.copy2(provenance_path,package/'reports/pretrain_provenance.json')
    shutil.copy2(ROOT/'reports/data_audit.json',package/'reports/data_audit.json')
    shutil.copy2(verification_log or ROOT/'reports/final_verification.log',package/'reports/verification.log')
    if not verify_runtime and (output/'packaged_local_runtime_check.json').exists():
        shutil.copy2(output/'packaged_local_runtime_check.json',package/'reports/packaged_local_runtime_check.json')
    with zipfile.ZipFile(package/'source_code.zip','w',zipfile.ZIP_DEFLATED) as archive:
        sources=list(ROOT.glob('*.py'))+list(ROOT.glob('*.md'))+list(ROOT.glob('requirements*.txt'))
        for folder in ['tools','tests','docs']:
            sources+=list((ROOT/folder).glob('*.py'))+list((ROOT/folder).glob('*.md'))+list((ROOT/folder).glob('*.mjs'))
        for file in sources:
            if file.name=='PROGRESS.md':
                continue
            archive.write(file,file.relative_to(ROOT))
    metric=selection['actual_single_image_validation']
    training_description=training_description or '所有微调候选从baseline_r18_224最佳权重初始化；复训应先按源码README完成基线，再按docs/OPTIMIZATION.md训练候选。'
    validation_description=validation_description or '训练仅使用官方train；valid用于探索性模型与checkpoint选择。测试集仅做冻结后的最终推理。'
    initialization_description=initialization_description or 'training_initialization中附已声明的官方train中间检查点，仅用于复训初始化，不由最终推理加载。'
    (package/'复现说明.md').write_text(f'''# 优化模型复现说明

固定方案：{selection['name']}。模型配置见 model/config.json，概率等权平均，阈值0.5。
{validation_description}
逐图黑边裁剪、等比例填充到各权重记录尺寸、ImageNet固定归一化；推理eval、无梯度、FP16。
原图逐张推理验证AUROC={metric['auroc']:.9f}，AP={metric['average_precision']:.9f}。
这是本地验证结果，不是官方测试分数；多次模型选择的偏差未被重采样区间消除。
本机100张验证图、5张预热、3轮batch1计时平均{selection['local_mean_ms']:.3f}ms。
尚未验证官方T4和官方Python3.12/PyTorch2.10.0环境，不能宣称官方100ms已通过。

在本目录运行：
python code/predict.py --data <glaucoma_first目录> --checkpoint model/config.json --output reproduced_submission.csv
python cost.py submission.csv <官方pictures_100> <官方pictures_5>

完整源码含训练/审计/测试在source_code.zip；依赖见code/requirements-local.lock.txt和官方目标requirements.txt。
本机CUDA依赖先安装：python -m pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 --index-url https://download.pytorch.org/whl/cu124
再运行：python -m pip install -r code/requirements-local.lock.txt
{training_description}
{initialization_description}
全部入选权重与固定融合配置已打包，直接推理不需要重新训练。
原始竞赛数据不包含在包内。预训练来源、许可证、SHA256详见reports及licenses。
由参赛者按真实队伍名称命名并自行上传。
''',encoding='utf-8')
    if verify_runtime:
        from verify_packaged_runtime import main as check_runtime
        check_runtime(output)
        shutil.copy2(output/'packaged_local_runtime_check.json',package/'reports/packaged_local_runtime_check.json')
    runtime=sum(p.stat().st_size for folder in ['model','code'] for p in (package/folder).rglob('*') if p.is_file())
    if runtime>500_000_000:
        raise ValueError('Runtime assets exceed 500MB')
    manifest=[dict(file=p.relative_to(package).as_posix(),bytes=p.stat().st_size,
                   sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(package.rglob('*')) if p.is_file()]
    save_json(package/'文件清单.json',dict(runtime_bytes=runtime,files=manifest))
    target=output/'optimized_materials.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(package.rglob('*')):
            if p.is_file(): archive.write(p,p.relative_to(package))
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None: raise ValueError('ZIP CRC verification failed')
        for item in manifest:
            if hashlib.sha256(archive.read(item['file'])).hexdigest()!=item['sha256']:
                raise ValueError('Archived hash mismatch')
    if target.stat().st_size>500_000_000: raise ValueError('Upload exceeds 500MB')
    save_json(output/'submission_validation.json',dict(rows=len(rows),names_and_labels_valid=True,
              runtime_bytes=runtime,zip_bytes=target.stat().st_size,zip_crc_and_hashes_valid=True,
              official_T4_verified=False))
    print('OPTIMIZED PACKAGE VERIFIED',target,flush=True)


if __name__=='__main__': main()
