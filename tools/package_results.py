"""Build reproducibility materials after completed training and frozen inference."""
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


def main():
    output=ROOT/'output'
    submission=output/'submission.csv'
    run=ROOT/'runs'/'baseline_r18_224'
    state=json.loads((run/'status.json').read_text(encoding='utf-8'))
    if state['state']!='complete':
        raise RuntimeError('Training is not complete')
    with submission.open(encoding='utf-8',newline='') as file:
        rows=list(csv.DictReader(file))
    names=test_names(ROOT/'data'/'raw'/'glaucoma_first')
    if [r['Image_name'] for r in rows]!=names:
        raise ValueError('Submission order/names differ from official test CSV')
    for row in rows:
        p=float(row['Probability'])
        if not 0<=p<=1 or int(row['Pred_Label'])!=int(p>=.5):
            raise ValueError('Invalid submission probability or label')
    package=output/'final_materials'
    if package.exists():
        raise FileExistsError('Preserve existing package; select a fresh package directory')
    (package/'code').mkdir(parents=True)
    (package/'model').mkdir()
    (package/'licenses').mkdir()
    (package/'pretrain_model').mkdir()
    (package/'reports').mkdir()
    for name in ['train.py','predict.py','model.py','data.py','utils.py','provenance.py','requirements.txt']:
        shutil.copy2(ROOT/name,package/'code'/name)
    shutil.copy2(ROOT/'cost.py',package/'code'/'cost.py')
    # Root cost entrypoint keeps the official CLI while importing code/predict.py.
    (package/'cost.py').write_text('import runpy,sys\nfrom pathlib import Path\nsys.path.insert(0,str(Path(__file__).resolve().parent/"code"))\nrunpy.run_module("cost",run_name="__main__")\n',encoding='utf-8')
    shutil.copy2(run/'best.pt',package/'model'/'best.pt')
    shutil.copy2(submission,package/'submission.csv')
    shutil.copy2(output/'模型核验材料与数据申明清单.xlsx',package/'模型核验材料与数据申明清单.xlsx')
    shutil.copy2(ROOT/'requirements-local.lock.txt',package/'reports'/'requirements-local.lock.txt')
    (package/'code'/'tools').mkdir()
    for name in ['prepare_data.py','bind_audit.py']:
        shutil.copy2(ROOT/'tools'/name,package/'code'/'tools'/name)
    shutil.copy2(ROOT/'reports'/'extraction.json',package/'reports'/'extraction.json')
    provenance=json.loads((ROOT/'reports'/'pretrain_provenance.json').read_text(encoding='utf-8'))
    for item in provenance:
        source=ROOT/item['path']
        if hashlib.sha256(source.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Provenance hash mismatch')
        shutil.copy2(source,package/item['path'])
    for name in ['pretrain_provenance.json','data_audit.json','gpu_smoke.json']:
        shutil.copy2(ROOT/'reports'/name,package/'reports'/name)
    for name in ['config.json','environment.json','best_metrics.json','history.jsonl']:
        shutil.copy2(run/name,package/'reports'/name)
    best=json.loads((run/'best_metrics.json').read_text(encoding='utf-8'))
    content=f'''# 参赛复现说明

模型：torchvision 官方 ResNet-18 ImageNet-1K 权重，有监督二分类微调。
训练使用官方训练集，验证集仅用于 checkpoint 选择，没有并入训练。
不使用外部医学数据、医学预训练、伪标签、测试统计量、TTA 或集成。
预处理：逐图黑边裁剪（阈值12）、保持比例填充224×224、ImageNet固定归一化。
推理：eval、无梯度、fp16、batch=1；输出阳性概率，类别阈值0.5。
最佳本地验证 AUROC：{best['auroc']:.8f}；average precision：{best['average_precision']:.8f}。
上述是验证集结果，不是官方测试成绩。

运行预测：python code/predict.py --data <解压后的glaucoma_first目录> --checkpoint model/best.pt --output submission.csv
官方计时：python cost.py submission.csv <pictures_100> <pictures_5>
完整训练源码及审计脚本另附 source_code.zip；按其中 README 操作，原始竞赛数据不包含在材料中。

本机训练环境和精度见 reports/environment.json。官方目标 requirements.txt 为 Python3.12 配套。
未完成官方 T4 环境复测，未收到官方100+5计时样本，不能声明100ms已通过。
权重和许可证的来源/哈希见 reports/pretrain_provenance.json。
材料请以队伍真实名称命名，由参赛者在平台自行上传。
'''
    (package/'复现说明.md').write_text(content,encoding='utf-8')
    manifest=[]
    for path in sorted(package.rglob('*')):
        if path.is_file():
            manifest.append({'file':path.relative_to(package).as_posix(),'bytes':path.stat().st_size,
                             'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    size=(package/'model'/'best.pt').stat().st_size
    if size>500_000_000:
        raise ValueError('Runtime weights exceed 500MB')
    save_json(package/'文件清单.json',{'runtime_model_bytes':size,'files':manifest})
    save_json(output/'submission_validation.json',{'rows':len(rows),'unique_names':len(set(names)),
              'probabilities_valid':True,'labels_consistent':True,'runtime_model_bytes':size,
              'official_T4_timing_verified':False})
    with zipfile.ZipFile(output/'baseline_materials.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package.rglob('*')):
            if path.is_file():
                archive.write(path,path.relative_to(package))
    with zipfile.ZipFile(output/'source_code.zip','w',zipfile.ZIP_DEFLATED) as archive:
        sources=list(ROOT.glob('*.py'))+list(ROOT.glob('*.md'))+list(ROOT.glob('requirements*.txt'))
        sources+=list((ROOT/'tools').glob('*.py'))+list((ROOT/'tests').glob('*.py'))+list((ROOT/'docs').glob('*.md'))
        for path in sources:
            archive.write(path,path.relative_to(ROOT))
    print('PACKAGED',str(output/'baseline_materials.zip'),flush=True)


if __name__=='__main__':
    main()
