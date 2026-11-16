import json
from pathlib import Path
from package_optimized import main
from v8_integrity import frozen,verify_evidence

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    output=ROOT/'output/v8';frozen(ROOT)
    selection=json.loads((output/'internal_selection.json').read_text());verify_evidence(selection['evidence'])
    gate=json.loads((output/'validation_ci_gate.json').read_text())
    if not gate['passed']:raise ValueError('No package without conservative validation filter')
    verify_evidence(gate['evidence'])
    provenance=json.loads((ROOT/'reports/pretrain_provenance_v5.json').read_text())
    provenance=[item for item in provenance if not item['path'].endswith('.pth') or Path(item['path']).name.startswith(('resnet50-','convnext_tiny-'))]
    target=output/'package_pretrain_provenance.json';target.write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    reports=[ROOT/'reports/v5_internal_split.json',ROOT/'reports/v5_internal_split_summary.json',ROOT/'reports/v5_capacity.json',
             ROOT/'reports/v8_latency_probe.json',output/'frozen_inputs.json',output/'internal_selection.json',
             output/'internal_predictions.npz',output/'validation_ci_gate.json']
    if (output/'frozen_inputs.initial.json').exists():reports.append(output/'frozen_inputs.initial.json')
    for name in ['v5_inner_convnext','v8_inner_color']:
        directory=ROOT/'runs'/name
        for filename in ['config.json','environment.json','best_metrics.json','history.jsonl','split_binding.json','status.json']:
            copy=output/(name+'_'+filename);copy.write_bytes((directory/filename).read_bytes());reports.append(copy)
    main(output,target,ROOT/'reports/v8_verification.log',
         training_description='R50固定沿用第三版两阶段官方train训练权重并水平翻转。ConvNeXt从公开ImageNet重新训练，配方color_robust：原H/V翻转及±15度旋转，颜色抖动(.30,.30,.25,.04)，灰度p.10，3x3高斯模糊sigma(.1,.8)、p.10。原内部train留出比较原图/亮度.70/饱和度.50的固定扰动，只允许清洁及最差AUROC均至少改善.0005的候选；冻结轮次及CN翻转策略后，从原始ImageNet全量重训8轮并固定保存该轮次，seed20261040。固定1:1概率平均，无融合权重或阈值拟合。详见OPTIMIZATION-v8.md。',
         extra_initialization=[(ROOT/'runs/v3_r50_384_w2/best.pt','r50_stage1_best.pt')],extra_reports=reports,
         verify_runtime=True,include_baseline_initialization=False,
         validation_description='只用官方train更新参数；复用train内部留出选择新CN的轮次及预声明资源合格翻转形态，并要求固定模拟颜色扰动下稳定性提升；R50不进入内部比较。全量officialvalid逐轮监控不选轮次，冻结组合最终对固定v6做一次升级检查，另要求AUROC增益至少.0005且事后描述性区间下界大于0。该区间未校正模型选择或患者聚类，模拟扰动不是外部验证，不能保证线上提升。测试集只做冻结后的最终预测。',
         initialization_description='R50阶段1检查点附training_initialization/r50_stage1_best.pt以供复训阶段2；新CN内外训练均直接从公开ImageNet初始化。公开预训练、原始来源/许可证/SHA、内部控制/实验配方和所需初始化链均记录，无外部医学数据或医学预训练。')
