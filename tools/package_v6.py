import json
from pathlib import Path
from package_optimized import main

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    provenance=json.loads((ROOT/'reports/pretrain_provenance_v5.json').read_text(encoding='utf-8'))
    provenance=[item for item in provenance if not item['path'].endswith('.pth') or Path(item['path']).name.startswith(('resnet50-','convnext_tiny-'))]
    target=ROOT/'output/v6/package_pretrain_provenance.json'
    target.write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    reports=[ROOT/'reports/v5_internal_split.json',ROOT/'reports/v5_internal_split_summary.json',
             ROOT/'reports/v5_capacity.json',ROOT/'reports/v5_latency_probe.json',
             ROOT/'output/v5/internal_selection.json',ROOT/'output/v5/selection_rejected.json',
             ROOT/'output/v6/input_fingerprints.json']
    for name in ['v5_inner_r50','v5_inner_convnext']:
        directory=ROOT/'runs'/name
        for file in ['config.json','environment.json','best_metrics.json','history.jsonl','split_binding.json','status.json']:
            copy=ROOT/'output/v6'/(name+'_'+file)
            copy.write_bytes((directory/file).read_bytes());reports.append(copy)
    main(ROOT/'output/v6',target,ROOT/'reports/v6_verification.log',
         training_description='固定两个成员均水平翻转并对概率等权1:1平均。R50沿用第三版两阶段官方train有监督训练；ConvNeXt为第五版内部选定架构/8轮/TTA后从公开ImageNet对官方train全量重训的第8轮。v5单模型未通过升级，完整结论随包，不隐瞒。此组合只在官方valid探索性检查，不能将v5内部留出结果当作组合的独立验证。详见docs/OPTIMIZATION-v6.md。',
         extra_initialization=[(ROOT/'runs/v3_r50_384_w2/best.pt','r50_stage1_best.pt')],
         extra_reports=reports,verify_runtime=True,include_baseline_initialization=False,
         validation_description='只用官方train更新参数；第五版train内部留出选ConvNeXt方案；此固定双模型组合只在官方valid做最终提升检查。官方valid曾多次用于探索选型，非新的独立验证。测试集只做冻结预测。',
         initialization_description='R50阶段1最佳检查点附training_initialization/r50_stage1_best.pt，用于第三版阶段2初始化；第三版阶段2已作为最终model成员，不重复保存。ConvNeXt从公开ImageNet初始化，内部划分/选择报告与两种公开权重均随包。')
