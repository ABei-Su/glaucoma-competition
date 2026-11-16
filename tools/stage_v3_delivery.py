"""Verify the final archive and stage the participant's third upload version."""
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from provenance import verify_prediction_binding


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(version='v3',label='第三版',reference='第二版97.97'):
    output=ROOT/'output'/version
    if json.loads((ROOT/'reports'/(version+'_finalization_status.json')).read_text())['phase']!='complete':
        raise ValueError('Finalization not complete')
    verify_prediction_binding(output/'submission.csv',output/'selected_config.json')
    validation=json.loads((output/'submission_validation.json').read_text())
    if validation['rows']!=5855 or not validation['zip_crc_and_hashes_valid']:
        raise ValueError('Submission checks failed')
    archive_path=output/'optimized_materials.zip'
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None: raise ValueError('ZIP CRC failed')
        manifest=json.loads(archive.read('文件清单.json'))
        for item in manifest['files']:
            if hashlib.sha256(archive.read(item['file'])).hexdigest()!=item['sha256']:
                raise ValueError('ZIP hash mismatch: '+item['file'])
        if archive.read('submission.csv')!=(output/'submission.csv').read_bytes():
            raise ValueError('Archive CSV mismatch')
        timing=json.loads(archive.read('reports/packaged_local_runtime_check.json'))
        if timing['mismatches'] or not timing['within_limit']: raise ValueError('Runtime check failed')
        for file,expected in timing['code_sha256'].items():
            if hashlib.sha256(archive.read(file)).hexdigest()!=expected:
                raise ValueError('Timing code binding mismatch')
        for item in timing['model_and_config']:
            source=Path(item['path'])
            relative=source.relative_to(output/'materials').as_posix()
            if hashlib.sha256(archive.read(relative)).hexdigest()!=item['sha256']:
                raise ValueError('Timing model binding mismatch')
    destination=ROOT.parent/(label+'提交_'+version)
    destination.mkdir(exist_ok=False)
    package_name=label+'_模型核验材料.zip'
    files={'submission.csv':'submission.csv','optimized_materials.zip':package_name,
           'submission_validation.json':'核对结果.json','selection_report.json':'验证结果.json',
           'bootstrap_comparison.json':'验证提升不确定性.json','packaged_local_runtime_check.json':'本机测速结果.json'}
    for source,name in files.items():
        shutil.copy2(output/source,destination/name)
        if digest(output/source)!=digest(destination/name): raise ValueError('Delivery copy mismatch')
    selection=json.loads((output/'selection_report.json').read_text(encoding='utf-8'))
    metric=selection['actual_single_image_validation']
    (destination/'上传与版本说明.txt').write_text(f'{label}固定方案：{selection["name"]}。\n评分入口上传本目录submission.csv；不要用Excel另存或修改列名。\n蓝色核验入口使用{package_name}，可按真实队伍名重命名。\n本地验证AUROC{metric["auroc"]:.8f}、AP{metric["average_precision"]:.8f}；提升不确定性见配套报告，不能保证平台上涨或达到99。\n本机包内100图、5预热、3轮batch1平均{timing["mean_ms"]:.6f}ms，预测不一致0张；官方T4及指定环境尚未复测。\n{reference}及原文件保留；本版平台分数未知。按平台当前允许时间上传，最终核验材料对应历史最高分CSV与实际模型。\n',encoding='utf-8')
    report=dict(rows=5855,archive_crc_and_hashes_verified=True,timing_model_code_binding_verified=True,
                csv_sha256=digest(destination/'submission.csv'),zip_sha256=digest(destination/package_name),
                zip_bytes=archive_path.stat().st_size,runtime_bytes=validation['runtime_bytes'],
                official_T4_verified=False)
    (destination/'交付校验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)
    print(str(destination),flush=True)


if __name__=='__main__': main()
