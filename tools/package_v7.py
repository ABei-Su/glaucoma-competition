import json
from pathlib import Path
from package_optimized import main as package_main
from provenance import copy_evidence_alias,digest
from utils import save_json

ROOT=Path(__file__).resolve().parents[1]

def main(output=None,version='v7'):
    output=Path(output) if output is not None else ROOT/'output/v7'
    provenance=json.loads((ROOT/'reports/pretrain_provenance_v7.json').read_text())
    provenance=[item for item in provenance if not item['path'].endswith('.pth') or Path(item['path']).name.startswith(('efficientnet_v2_s-','convnext_tiny-'))]
    target=output/'package_pretrain_provenance.json';target.write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    reports=[ROOT/'reports/v5_internal_split.json',ROOT/'reports/v5_internal_split_summary.json',
             ROOT/'reports/v7_capacity.json',ROOT/'reports/v7_latency_probe.json',output/'internal_selection.json']
    for report in [output/'deployment_revision.json',output/'independent_validation_check.json',ROOT/'reports/v7_latency_diagnosis.json',ROOT/'reports/v7_fusion_probe.json']:
        if report.exists():reports.append(report)
    if version=='v7b':
        revision=json.loads((output/'deployment_revision.json').read_text())
        expected={item['path']:item['sha256'] for item in revision['evidence']}
        for path,sha256 in expected.items():
            if digest(path)!=sha256:raise ValueError('Frozen deployment evidence changed before packaging')
        aliases=[(ROOT/'output/v7/candidate_00.json','v7_unfused_candidate.json'),
                 (ROOT/'output/v7/local_timing_candidates.json','v7_unfused_local_timing_candidates.json'),
                 (ROOT/'reports/v7_selection.log','v7_selection_failure.log'),
                 (ROOT/'reports/v7_finalization_status.json','v7_finalization_failure.json')]
        records=[]
        for source,name in aliases:
            target_copy=output/name
            records.append(copy_evidence_alias(source,target_copy,expected[str(source)]))
            reports.append(target_copy)
        evidence_map=output/'v7_failure_evidence_map.json';save_json(evidence_map,records);reports.append(evidence_map)
    for name in ['v7_inner_efficientnet','v5_inner_convnext']:
        directory=ROOT/'runs'/name
        for filename in ['config.json','environment.json','best_metrics.json','history.jsonl','split_binding.json','status.json']:
            target_copy=output/(name+'_'+filename)
            target_copy.write_bytes((directory/filename).read_bytes());reports.append(target_copy)
    fallback_note=('原始双模型翻转方案及卷积融合试验均超时，按OPTIMIZATION-v7b.md固定采用原内部排序第二的无翻转双模型；未使用卷积融合，原检查点未改写。' if version=='v7b' else '')
    package_main(output,target,ROOT/'reports'/(version+'_verification.log'),
         training_description='第七版EfficientNetV2-S从官方ImageNet开始，在固定train内部留出选轮次与部署形态，再从原始公开权重对官方train全量重训8轮，固定保存内部选择轮次。若入选双模型，则与v5_full ConvNeXt按固定1:1概率平均，两成员采用相同翻转策略。没有融合比例或阈值拟合，详见OPTIMIZATION-v7.md。'+fallback_note,
         extra_reports=reports,verify_runtime=True,include_baseline_initialization=False,
         validation_description='只用官方train更新参数；复用train内部15%留出选择EfficientNet轮次及单/双模型和翻转形态；全量训练逐轮记录officialvalid指标，但固定保存内部选择轮次，不以这些指标改参数或选轮；冻结候选再对第六版做最终提升检查。两个验证来源均已用于探索，不能称新独立验证。',
         initialization_description='新EfficientNet内部候选与全量重训均从公开ImageNet初始化；若使用ConvNeXt成员，其v5全量配方来自内部选择，亦从公开ImageNet初始化。两种公开权重、来源SHA、内部划分和选择报告随包。无旧比赛模型初始化链依赖。')


if __name__=='__main__':main()
