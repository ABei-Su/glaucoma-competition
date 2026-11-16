from pathlib import Path
import json
from package_optimized import main

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    selection=json.loads((ROOT/'output/v4/selected_config.json').read_text(encoding='utf-8'))
    architectures={json.loads(((ROOT/'output/v4'/entry['checkpoint']).parent/'config.json').read_text(encoding='utf-8')).get('architecture','resnet18') for entry in selection['models']}
    provenance=json.loads((ROOT/'reports/pretrain_provenance_v3.json').read_text(encoding='utf-8'))
    provenance=[item for item in provenance if not item['path'].endswith('.pth') or Path(item['path']).name.startswith('resnet50-') or 'resnet18' in architectures]
    provenance_path=ROOT/'output/v4/package_pretrain_provenance.json'
    provenance_path.write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    main(ROOT/'output/v4',provenance_path,ROOT/'reports/v4_verification.log',
         training_description='第四版两个R50候选均从第三版v3_r50_384_w1最佳权重独立继续有监督微调，分别448/6轮和384/4轮。第三版第一阶段从官方ImageNet-1K V2开始、第二阶段有监督续训。完整配置见报告及docs/OPTIMIZATION-v4.md。初始化链权重附training_initialization，不被最终推理调用。',
         extra_initialization=[(ROOT/'runs/v3_r50_384_w2/best.pt','r50_stage1_best.pt'),
                               (ROOT/'runs/v3_r50_384_w1/best.pt','v3_r50_best.pt')],
         extra_reports=[ROOT/'reports/r50_capacity.json',ROOT/'reports/r50_latency_probe.json',
                        ROOT/'reports/v4_capacity.json',ROOT/'reports/v4_latency_probe.json'],verify_runtime=True,
         include_baseline_initialization='resnet18' in architectures)
