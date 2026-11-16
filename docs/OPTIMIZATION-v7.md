# 第七版新架构与固定组合计划

目标：继续探索更强泛化模型，用户平台目标99.3。第六版用户截图98.60只记提交历史，不作拟合、权重调参或候选选择输入。全部实验为探索性比赛建模，不能保证目标分数。

固定新架构：torchvision EfficientNetV2-S，官方公开ImageNet-1K V1权重，BSD许可证。官方输入384与ImageNet均值/标准差；继续沿用当前逐图确定性裁剪/等比例填充、训练翻转/旋转/温和颜色增强。无外部医学数据、伪标签、测试统计拟合、年龄或患者身份特征。

资源优先：合成训练batch4/8/12/16，固定contiguous，实测峰值<5GB者选择吞吐最快批量；先测100张valid、5预热、3轮batch1原图延迟，仅看耗时不按探测预测选模型。评估4种预声明部署形态（EfficientNetV2-S单独plain/TTA，与固定v5_full ConvNeXt等权组合plain/TTA），推理速度<=100ms的形态才进入后续内部比较。若全部超限，停下记录，不临时追加架构。没有新的生产依赖。

内部候选v7_inner_efficientnet：只用固定reports/v5_internal_split.json的43686训练/7710留出（249阳性）；从公开ImageNet开始，不用任何旧比赛权重初始化。384、8轮、lr1e-4、pos_weight1、seed20261033、batch按容量选择、AdamW weight_decay1e-4/cosine horizon8、最低lr1e-5、FP16前向FP32 sigmoid。按留出AUROC优先、AP同分辅助选best。

内部比较：新EfficientNet best与已完成且对相同内部留出无训练接触的v5_inner_convnext/best.pt。只评估速度合格形态：新模型plain/TTA、两模型等权plain/TTA，最多4个新候选；原ConvNeXt_TTA为固定内部参考。内部分别用同一原图路径/无损缓存batch1，核对名字标签、来源及SHA。没有拟合融合比例、阈值或追加种子。不使用v3等已见过内部留出的模型参与该内部验证。

若最高新候选AUROC严格高于固定内部ConvNeXt_TTA（完全同分时AP提高），冻结新架构最佳轮次E、plain/TTA及是否双模型；否则本计划不做全量重训、不生成测试CSV，保留失败结果。

通过内部门槛后，v7_full_efficientnet从原始公开ImageNet重新训练官方train全量8轮，seed20261034，lr/augmentation/batch/horizon与内部相同，固定checkpoint_epoch=E。若选择双模型，第二成员使用已完成v5_full（其8轮训练配方对应内部ConvNeXt方案）。沿用训练器逐轮记录officialvalid指标作监控，固定保存E且不据这些指标改参数；不再用officialvalid挑轮次或形态。

唯一冻结最终候选按完整原图batch1对照固定第六版output/v6/selected_config.json；严格AUROC提高（完全同分AP提高），并通过最终本机包内<=100ms、运行权重/必要文件与ZIP均<=500MB检查后，才交付可上传的CSV及材料。官方valid已多次用于探索检查，不能称独立测试。内部划分也曾用于v5，不是新留出的确认样本；无患者ID，不能排除同患者或近重复依赖，区间仅描述。

第六版及原有模型/原始数据不可覆盖，全部新文件位于runs/v7_*、output/v7、reports/v7_*。只在模型冻结并通过官方valid升级检查后生成5855张test预测。最终上传由用户完成；官方T4/指定100+5图片/环境另需验证。

实施验收：架构二分类/预训练加载/冻结重载测试先失败后通过；保持旧模型与断点兼容；完整verify及JS语法检查；训练/内部选择/全量配方和证据指纹冻结；原图对照、CSV格式、包内cost、CRC/SHA、模型与代码绑定、声明渲染及独立交付核验。

2026-10-26独立复核后的流程澄清（首次v7测试预测之前）：原文“最终包检查通过才输出CSV”与打包必须包含CSV的实际依赖不一致。修正为：模型冻结且通过官方valid升级门槛后，在output/v7生成供打包核验的暂存CSV；包内测速、体积及完整性检查全部通过且finalization状态complete后，才复制到用户交付目录。后续步骤失败时，暂存CSV及其指纹文件改名为submission.unverified.*并在失败状态注明，禁止交付。该澄清不改变数据边界、候选、超参数、轮次或升级门槛。

恢复保护：跳过已完成训练前必须核对完整请求配方、缓存/划分指纹、checkpoint及三个轮次指标记录；全量批量来自内部冻结配方。容量探测、延迟探测、预训练来源、固定完整ConvNeXt成员和第六版参考的内容指纹在内部比较开始时冻结，最终验证前再次校验。
