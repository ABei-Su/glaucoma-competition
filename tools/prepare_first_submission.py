"""Verify and stage the frozen baseline without using current optimized code."""
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(content):
    return hashlib.sha256(content).hexdigest()


def main():
    output=ROOT/'output'
    source=output/'final_materials'
    manifest=json.loads((source/'文件清单.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path=source/item['file']
        if path.stat().st_size!=item['bytes'] or sha(path.read_bytes())!=item['sha256']:
            raise ValueError('Frozen baseline material changed: '+item['file'])
    submission=output/'submission.csv'
    if submission.read_bytes()!=(source/'submission.csv').read_bytes():
        raise ValueError('Frozen submission differs from baseline package')
    with submission.open(encoding='utf-8',newline='') as file:
        reader=csv.DictReader(file)
        if reader.fieldnames!=['Image_name','Probability','Pred_Label']:
            raise ValueError('Wrong CSV header')
        rows=list(reader)
    with (ROOT/'data/raw/glaucoma_first/test.csv').open(encoding='utf-8-sig',newline='') as file:
        expected=[r['Image_name'] for r in csv.DictReader(file)]
    if len(rows)!=5855 or len(set(expected))!=5855 or [r['Image_name'] for r in rows]!=expected:
        raise ValueError('Test filenames/count/order mismatch')
    for row in rows:
        p=float(row['Probability'])
        if not 0<=p<=1 or row['Pred_Label']!=str(int(p>=.5)):
            raise ValueError('Invalid probabilities or labels')
    for name in ['baseline_materials.zip','source_code.zip']:
        with zipfile.ZipFile(output/name) as archive:
            if archive.testzip(): raise ValueError('ZIP CRC failed')
    destination=ROOT.parent/'首次提交_v1'
    destination.mkdir(exist_ok=False)
    shutil.copy2(submission,destination/'submission.csv')
    archive_path=destination/'第一版_模型核验材料.zip'
    with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in source.rglob('*'):
            if path.is_file(): archive.write(path,path.relative_to(source))
        archive.write(output/'source_code.zip','source_code.zip')
        archive.writestr('环境安装补充.txt','本机训练环境复现：先执行 python -m pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 --index-url https://download.pytorch.org/whl/cu124，再按 reports/requirements-local.lock.txt 安装其余依赖。官方T4及官方环境尚未复测。\n')
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip(): raise ValueError('Final ZIP CRC failed')
        for item in manifest['files']:
            if sha(archive.read(item['file']))!=item['sha256']:
                raise ValueError('Final ZIP material hash mismatch')
    report=dict(version='frozen baseline v1',rows=len(rows),columns_valid=True,names_order_valid=True,
                probabilities_labels_valid=True,all_frozen_material_hashes_verified=True,zip_crc_verified=True,
                submission_sha256=sha(submission.read_bytes()),zip_bytes=archive_path.stat().st_size,
                validation_auroc=0.9828811172245572,official_test_score=None,official_T4_verified=False)
    (destination/'核对结果.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (destination/'上传说明.txt').write_text('1. 评分用文件：submission.csv。点击赛题页面橙色“上传文件”入口，选择该CSV。不要用Excel另存，不要改列名和文件名。\n2. 合规核验材料：第一版_模型核验材料.zip。用于蓝色“模型核验材料与数据申明清单”入口；可按真实队伍名重命名为“队伍名_最终提交材料.zip”。不要把它当评分CSV上传。\n3. 截图显示成功出分后当天不能继续提交，次日可提交改进版；当天报错的额外次数以页面实时提示为准。不要为了次日更新而使用撤销入口。\n4. 保留提交记录/版本/成绩。最终核验包中的CSV应对应历史最高分版本；若后续换版本，模型和材料也要同步更新。\n5. 本版验证AUROC为0.98288112，不是官方测试成绩。官方T4及100ms尚未复测。\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)
    print(str(destination),flush=True)


if __name__=='__main__': main()
