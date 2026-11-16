# 青光眼二分类参赛项目

## 第十八版单一组合试验
用户批准的最终独立候选：fresh R50、384、BCEpos1、固定16轮/epoch16，与旧CN8水平TTA等权，仅一份交付CSV。方案 `docs/science-superpowers/plans/2026-11-05-v18-r50-longer.md`；入口 `tools/launch_v18.ps1`，训练6小时上限，成功后自动一次4小时finalizer。`reports/v18_launch.json`、`v18_exit.json`、`v18_finalization_status.json`记录阶段；`awaiting_review`并不表示已通过独立交付核验。不得重复启动、自动恢复、清预约或改冻结配方。旧98.60模型/CSV/已上传材料保持；新CSV不能与旧核验包混配，线上提交资格及官方硬件尚未验证。

## 第十七版固定阳性权重实验
方案见 `docs/science-superpowers/plans/2026-10-30-v17-positive-weight.md`。保持第十五版fresh R50配方，仅把BCE阳性训练权重1改为2，固定8轮及第8轮检查点；新单模型及固定CN8等权组合直接对照第十五版原配置。全部旧模型、提交和明日备份保留。

入口 `tools/launch_v17.ps1` 限3小时训练，成功后自动一次4小时最终验证/交付。已有预约、run或终态不可覆盖，不自动重试。监督 `reports/v17_supervision.json`；独立核验之前不声明新CSV可提交。验证集重复使用，结果属探索性，线上及官方T4尚未验证。

## 第十六版固定低学习率延伸实验
方案见 `docs/science-superpowers/plans/2026-10-30-v16-refinement.md` 与 `docs/ADR-016-r50-refinement.md`。从第十五版R50固定第8轮初始化新优化器，再训练4轮，锁定第4轮；单模型及固定CN8等权组合分别直接对照第十五版原配置。旧训练/推理源码与第十五版交付均保留，全部新产物E盘。

`tools/launch_v16.ps1` 有原子预约、冻结来源及独立审查门控，训练限3小时，成功后自动一次4小时 `tools/launch_v16_finalizer.ps1`；禁止重复启动或自动恢复。进度与终态在 `reports/v16_supervision.json`、`v16_exit.json`、`v16_finalization_status.json`。最终CSV在独立CPU核验与审查前不标记可提交；本地指标不是线上成绩或官方T4核验。

## 第十三版三折 Mixup 实验
完整计划见 `docs/science-superpowers/plans/2026-10-29-breakthrough.md`，契约见 `docs/ADR-013-mixup-folds.md`。
独立的 `train_mixup.py` 与 schema2 拆分验证保持旧训练器不变；从公开权重分别训练三个互斥留出折的 BCE 控制和唯一 Mixup 候选，固定第8轮，按三折 AUROC 与 OOF AP 的预定门槛决定是否全量训练。
独立审查与 `python tools/verify.py` 通过后，`python -m tools.run_v13` 顺序执行一次资源预检、六次拟合和条件全量交付。总比较预算12小时，晋级后额外4小时；禁止重复启动已有状态/目录。
状态为 `reports/v13_status.json`，子进程证据为 `reports/v13_child.json`。`awaiting_review` 表示仅已暂存交付，独立核验记录批准后才由 `python -m tools.run_v13 --stage accept_delivery` 转为完成。资源或科学门槛失败不会生成新的试交文件；旧最高分版本保留。不保证99.3，官方T4尚未验证，用户自行上传。

## 第十二版固定Focal训练实验
方案见 `docs/OPTIMIZATION-v12.md`、`docs/ADR-012-focal-training.md`。在原384ConvNeXt配方上固定 gamma2/alpha-disabled/FP32mean Focal Loss，与原BCE_TTA控制做同内部划分对比；`focal_loss.py` 与 `train_focal.py` 新文件保存loss/public来源和严格断点合约，不修改旧训练/预测模块。
完整verify和独立审查后，`python -m tools.run_v12` 先做一次真实容量/速度探测，再执行内部训练与条件全量重训；`python -m tools.finalize_v12` 有限等待并完成固定验证、身份/资源检查和唯一试交CSV。运行前将本次完整验证日志保存为 `reports/v12_verification.log`，已有v12目录/状态/锁时禁止重复启动或自动清理。状态见 `reports/v12_status.json`、`reports/v12_finalization_status.json`；全部新产物E盘，用户自行上传，不保证99.3。

## 第十一版448分辨率实验
原v11双TTA组合一次实测102.63ms超出100ms，已在监督训练前终止并保留记录，见 `docs/RESULTS-v11.md`。独立v11b固定保留R50翻转、CN448使用plain；新方案见 `docs/OPTIMIZATION-v11b.md` 和 `docs/ADR-011-v11b-resource-shape.md`，新源的可审计派生见 `reports/v11b_derivation.json`。对应运行入口 `tools.run_v11b`、`tools.finalize_v11b`，状态和产物使用v11b独立命名。不会重跑同一被拒绝形态直到过线。
固定问题与方案见 `docs/science-superpowers/questions/2026-10-28-convnext448.md`、`docs/OPTIMIZATION-v11.md` 和 `docs/ADR-010-v11-resolution.md`。复用原模型/训练/预测代码与448无损缓存，先测固定双模型速度和batch16显存，再比较同划分新448与原384控制。内部合格才从公开权重全量重训，固定轮次和等权融合不按官方valid重新选择。
先运行完整 `python tools/verify.py` 保存到新的 `reports/v11_verification.log` 并完成独立源码审查；`python -m tools.run_v11` 有限执行资源探测、内部比较与条件全量训练，`python -m tools.finalize_v11` 有限等待并固定验证/生成试交CSV。`tools/predict_v11.py` 显式固定后端和模型/代码指纹。已有v11产物或锁时禁止重复启动，需先审计；不会自动清理锁或覆盖旧版。状态见 `reports/v11_status.json` 和 `reports/v11_finalization_status.json`。

## 第十版图像与年龄实验
固定方案见 `docs/OPTIMIZATION-v10.md`，schema2和缺失年龄约定见 `docs/ADR-009-age-input-contract.md`。
新文件 `clinical.py`、`train_age.py`、`clinical_predict.py` 独立支持图像加年龄，不修改旧版训练/预测模块。
Age均值和尺度只由实际训练行拟合，内部留出/官方验证/测试年龄不参与拟合。使用现有384缓存、8轮、16batch和原内部划分；训练从公开ImageNet初始化。
`tools/probe_v10.py` 先测显存和固定双模型速度，`tools/run_v10.py` 冻结来源后顺序完成内部训练、真实/全部缺失年龄对照，并在达标后自动按固定轮次全量重训。运行前保存 `python tools/verify.py` 至 `reports/v10_verification.log`；来源和元数据哈希变化会拒绝继续。
状态见 `reports/v10_status.json`；中断恢复必须先核实PID/锁和冻结配方，不自动清理锁或覆盖终态。新模型需配套schema2配置和仅含Image_name/Age的元数据表，旧Predictor不能加载；用户仍自行上传结果。

2026-10-28 第一次训练在首轮完成前中断，没有完整检查点。单次显式恢复见 `docs/RECOVERY-v10.md`：`tools/recover_v10.py` 只在两原进程均已退出、原配方/冻结来源不变且无检查点时归档并重启原首轮；独立审查见 `docs/REVIEW-v10-recovery.md`。`tools/finalize_v10_age.py` 有限等待全量训练合格终态，按 `docs/FINALIZATION-v10.md` 固定验证和生成试交CSV；失败不会把半成品目录当作交付。

## 第九版固定权重平均实验
方案与兼容性约定见 `docs/OPTIMIZATION-v9.md`、`docs/ADR-008-fixed-weight-soup.md`。
`tools/compare_v9_internal.py` 只比较原 mild ConvNeXt 与 v8 color-robust ConvNeXt 的固定50:50权重平均，固定水平翻转TTA；先验证初始化/数据/轮次匹配，再在原官方train内部留出集做原图与两种固定颜色扰动评估。
先运行 `python tools/verify.py` 并保存日志 `reports/v9_verification.log`，再运行比较脚本。结果写 `output/v9`、`reports/v9_status.json`；未通过门槛就停止，不生成测试CSV。已有 `experiment.lock` 必须先确认PID结束再人工归档，禁止重复启动。派生模型不是训练检查点，不能用于 `--resume`。全部旧交付保留。

仅供本次获授权的比赛使用。原始数据与 zip 不进入 Git、不上传外部服务。

## 基线方案（已完成）
ResNet-18，公开 ImageNet-1K 权重，有监督微调；224×224，逐图黑边裁剪并保持比例填充；训练翻转、小角度旋转与温和颜色增强。BCEWithLogitsLoss 的阳性权重为训练阴阳数量比的平方根，AdamW，6 epochs，cosine LR。保留官方验证集，以 AUROC 首选模型，average precision 同分辅助，同时报告梯形 AUPRC。未使用年龄，未使用医学预训练、外部医学数据、伪标签、测试统计量、TTA 或集成。

## 命令（在本目录执行）
```powershell
python tools/verify.py
python tools/fetch_pretrain.py
python tools/extract_data.py
python tools/prepare_data.py
python train.py
python predict.py
python cost.py output/submission.csv <official_100_directory> <official_5_directory> --checkpoint runs/baseline_r18_224/best.pt
```

解压命令不覆盖已有文件，不应重复运行。训练中断后用 `python train.py --resume` 从上一完整轮恢复。非正常杀进程遗留的 `training.lock` 应先核对 PID 已结束，再手动移除；勿同时运行同一 run。正式训练输出为 `runs/baseline_r18_224`。原始数据 `data/raw`，训练/验证缓存 `data/cache224`，结果 `output`。

官方环境为 Python3.12、torch2.10.0、torchvision0.25.0、CUDA12.8/cuDNN9，T4 16GB。本机现有环境为 Python3.9.13、torch2.4.1+cu124、torchvision0.19.1+cu124，RTX4060 Laptop 8GB；可移植代码不等于官方环境已经验证。详见 reports 和 run 内的环境、指标记录。

## 提交
用户自行上传，不自动访问参赛账号。CSV 字段 `Image_name,Probability,Pred_Label`；UTF-8；5855 个唯一文件名；阈值0.5。预测概率按单图 fp16 推理，cost.py 复用相同 Predictor。官方 100+5 图片未提供前不冒充完成官方测速；本机性能只作参考。模型与必要推理文件解压后≤500MB，上传包≤页面500M限制。

来源与权重 SHA256：reports/pretrain_provenance.json；许可证：licenses/。精度、验证集用途、预处理、环境和权重都必须随最终核验材料声明。

## 优化流程
基线验证 AUROC 0.9828811172，最佳第5轮；基线提交与材料保留在 output/。
候选与边界见 docs/OPTIMIZATION.md：224/384继续有监督微调、不同阳性权重、固定水平翻转TTA、最多两模型等权平均。
候选只在官方valid比较，不使用测试预测作选择依据。优化结果生成后位于 output/optimized/，不覆盖基线。

```powershell
python tools/prepare_resolution.py --size 384 --workers 6
python train.py --run runs/opt_r18_224_w2 --epochs 8 --lr 0.0001 --pos-weight 2 --init-checkpoint runs/baseline_r18_224/best.pt
python tools/continue_optimization.py
python tools/select_optimized.py
python tools/bootstrap_comparison.py
python predict.py --checkpoint output/optimized/selected_config.json --output output/optimized/submission.csv
node tools/build_checklist.mjs --optimized
python tools/package_optimized.py
```

首次缓存准备与224训练可并行；continue_optimization等待两者完成后顺序训练384候选，避免GPU争用。
已有READY缓存不重复生成；训练完成的run不重复启动。XLSX生成器需要 @oai/artifact-tool，本机使用Codex bundled runtime。
优化推理JSON格式见 docs/ADR-002.md。官方三参数cost入口由材料包根目录cost.py自动绑定该包model/config.json。
缓存PNG压缩等级6为无损，磁盘剩余不足2GB会停止准备，防止写满；不会修改原始图像。

## 第三版实验
详细固定方案见 docs/OPTIMIZATION-v3.md，架构兼容约定见 docs/ADR-003.md。
train.py新增 --architecture resnet50，旧参数及旧ResNet18检查点兼容。ResNet50使用官方ImageNet-1K V2，非医学预训练。
先运行tools/fetch_r50.py、tools/probe_r50.py和tools/probe_r50_latency.py，再执行tools/run_v3.py。
tools/finalize_v3.py --node <node.exe路径>等待训练后，使用固定v2作严格对照；只有验证路径改善才生成新的测试CSV和材料。
所有第三版产物位于output/v3与runs/v3_*，前两版保留。训练中断按同样参数--resume；清理残留锁前必须确认旧PID已结束。

## 第四版实验
第三版平台截图98.51仅作提交历史，固定验证对照为output/v3/selected_config.json。
第四版计划见docs/OPTIMIZATION-v4.md：从第三版best独立继续448/6轮和384/4轮，比较20个固定候选。
先运行tools/probe_v4.py和tools/prepare_resolution.py --size 448，再运行tools/run_v4.py。
tools/finalize_v4.py --node <node.exe路径>完成验证、相同原图路径严格对照、冻结测试预测与含测速证据的材料。
第三版已交付目录不覆盖。第四版结果只写output/v4，只有验证改善才生成新CSV。

## 第五版实验
固定问题、比较和资源门限见docs/OPTIMIZATION-v5.md，数据边界见docs/ADR-004-v5-internal-validation.md。
新增ConvNeXt-Tiny官方ImageNet权重，先运行tools/fetch_convnext.py、tools/prepare_v5_split.py、tools/probe_v5.py。
tools/run_v5.py顺序训练两个全新内部候选，只用官方train内部留出选架构/轮次/TTA，再以固定轮次做官方train全量重训。
--split-manifest禁止旧有监督检查点初始化；--checkpoint-epoch锁定要保存的轮次，默认0维持原行为。
第五版单独写runs/v5_*、output/v5；第三版最高已提交98.51保留。验证提升不保证平台达到用户目标99.3。

## 第六版固定组合
第五版单模型未通过第三版验证对照，结论见docs/RESULTS-v5.md，未生成第五版测试CSV。
第六版仅测试预先写定的一种组合：第三版R50_TTA与第五版ConvNeXt_TTA的1:1概率平均；无权重拟合。
tools/prepare_v6.py核对已保存原图验证预测并生成唯一候选；tools/finalize_v6.py执行完整原图对照、测速和达标后打包。
固定方案与限制见docs/OPTIMIZATION-v6.md。产物独立位于output/v6，文件核验通过后由用户自行上传。

## 第七版新架构实验
固定方案见docs/OPTIMIZATION-v7.md，兼容约定见docs/ADR-005-efficientnet-v2.md。
tools/fetch_v7.py校验EfficientNetV2-S公开ImageNet权重；tools/probe_v7.py实测容量及四种部署形态速度。
tools/run_v7.py仅使用速度合格形态在复用train内部留出比较，改善后冻结配方并从公开权重全量重训。
tools/finalize_v7.py与固定第六版做原图对照，达标后才生成测试CSV与核验材料；内部未改善会直接停止。
所有第七版文件在runs/v7_*、output/v7、reports/v7_*，第六版及既有交付保留。

原v7双模型翻转方案最终本机测速108.90ms超限；独立Conv/BN融合试验103.34ms仍超限，均未交付。资源回退v7b固定采用原内部排序第二的无翻转双模型，复用原epoch6权重，单独完整验证和打包，见docs/OPTIMIZATION-v7b.md。仅finalization complete且独立交付校验通过后才可使用第七版文件。

## 第八版固定颜色稳定性实验
第七版平台用户反馈98.46，仅记历史；第六版98.60及原文件保留。v8计划见docs/OPTIMIZATION-v8.md，新增训练增强profile见ADR-007。训练默认mild保持旧行为；color_robust只对训练生效，断点恢复不得切换profile。tools/run_v8.py执行资源门槛、匹配既有CN控制的单个增强候选、原图及固定颜色扰动内部筛选、达标后固定轮次全量重训。tools/finalize_v8.py对固定v6完整验证，另做预声明描述性增益区间过滤；失败不生成测试CSV，成功才打包。所有新模型/报告/输出独立使用v8前缀，不以隐藏测试或排行榜拟合。不能保证平台99.3。
