from pathlib import Path
from package_optimized import main

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    main(ROOT/'output/v3',ROOT/'reports/pretrain_provenance_v3.json',ROOT/'reports/v3_verification.log',
         training_description='R50第一阶段从torchvision官方ImageNet-1K V2初始化，第二阶段从第一阶段best继续有监督微调。R18沿用前两版。完整配置见各run报告及docs/OPTIMIZATION-v3.md；未使用外部医学数据。R50第二阶段起点附training_initialization/r50_stage1_best.pt。',
         extra_initialization=[(ROOT/'runs/v3_r50_384_w2/best.pt','r50_stage1_best.pt')],
         extra_reports=[ROOT/'reports/r50_capacity.json',ROOT/'reports/r50_latency_probe.json'],verify_runtime=True)
