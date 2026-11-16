import json
from pathlib import Path
from package_optimized import main

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    provenance=json.loads((ROOT/'reports/pretrain_provenance_v5.json').read_text(encoding='utf-8'))
    provenance=[item for item in provenance if not item['path'].endswith('.pth') or Path(item['path']).name.startswith(('resnet50-','convnext_tiny-'))]
    target=ROOT/'output/v5/package_pretrain_provenance.json'
    target.write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    reports=[ROOT/'reports/v5_internal_split.json',ROOT/'reports/v5_internal_split_summary.json',
             ROOT/'reports/v5_capacity.json',ROOT/'reports/v5_latency_probe.json',ROOT/'output/v5/internal_selection.json']
    for name in ['v5_inner_r50','v5_inner_convnext']:
        directory=ROOT/'runs'/name
        # Unique output names avoid collisions while retaining both internal-run evidence sets.
        for file in ['config.json','environment.json','best_metrics.json','history.jsonl','split_binding.json','status.json']:
            copy=ROOT/'output/v5'/(name+'_'+file)
            copy.write_bytes((directory/file).read_bytes());reports.append(copy)
    main(ROOT/'output/v5',target,ROOT/'reports/v5_verification.log',
         training_description='第五版先从公开ImageNet权重独立训练ResNet50和ConvNeXt-Tiny，仅官方train内部15%留出选架构、轮次E与TTA；再从所选原始公开权重对官方train全量重新训练8轮，固定保存E轮。种子、划分、选择和训练配置见配套报告及docs/OPTIMIZATION-v5.md。',
         extra_reports=reports,verify_runtime=True,include_baseline_initialization=False,
         validation_description='只用官方train更新参数。内部留出用于选架构/轮次/TTA；官方valid用于冻结候选对第三版的最终提升检查，不重新挑选轮次或架构。测试集只做冻结预测。',
         initialization_description='两个内部候选和最终重训均从随包公开ImageNet权重初始化；没有使用旧比赛检查点。内部留出清单见reports/v5_internal_split.json，完整源码和固定轮次参数可复现选择。')
