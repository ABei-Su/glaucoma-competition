# 第五版独立验证与架构比较计划

问题：只使用官方训练数据，从公开ImageNet初始化的ConvNeXt-Tiny与ResNet50，在训练集内部固定留出样本上比较后锁定架构、训练轮次和翻转策略，能否在已使用的官方valid上复现提升，并满足提交体积和速度要求。用户目标平台99.3，仅作为最终目标，不作拟合信号。

这是探索性比赛优化，不是确认性临床研究。第四版平台退步具体主因未知；新增内部划分是方法改进，不能声称已经证实分布差异。原有模型已见过官方train全部样本，禁止作为内部留出的无泄漏对照。

固定数据方案：仅官方train按True_Label分层，每类按SHA256('20261029|'+Image_name)排序留出round(n*0.15)，其余训练。保存完整名字清单、CSV SHA及清单SHA，两个架构使用同一划分。确认名字互斥且完整覆盖train；official valid/test不能进入内部train/holdout。现有原图SHA审计无跨集精确重复。无患者标识，不能排除近重复或同患者相关性，内部划分不是患者独立验证。不能将曾反复使用的官方valid称为新holdout。

固定比较2个新run，均从各自官方ImageNet权重开始，不加载任何旧比赛微调权重：
- v5_inner_r50：384、8 epochs、lr2e-4、pos_weight1、seed20261030。
- v5_inner_convnext：384、8 epochs、lr1e-4、pos_weight1、seed20261031。
AdamW weight_decay1e-4，cosine horizon8、最低lr=initial/10；原有翻转/旋转/温和颜色增强；FP16前向/FP32 sigmoid。固定contiguous内存格式。批量仅以合成容量测量在8/12/16中选吞吐最快且峰值<5GB者；不用标签作硬件决策。单图原图延迟仅用官方valid前105图测容量，不将探测预测用于选型。

两个run只用内部holdout做逐轮验证，AUROC优先、AP同分辅助保存best。然后对各自best在内部holdout单图评估plain及水平翻转平均，4个候选按AUROC/AP排序，锁定第一名架构、该架构best轮次E与TTA。无融合权重拟合、阈值搜索、追加种子或追加候选。

最终v5_full从所选架构的原始ImageNet权重重新训练官方train全量8轮，seed20261032，学习率与内部run相同、cosine horizon8；预先锁定checkpoint-epoch=E，仅保存E轮为推理best，后续valid结果不改变已选轮次。内部holdout回归训练，officialvalid仍不并入训练。只对固定最终checkpoint/TTA与固定第三版做官方valid单图原图对照。严格AUROC提升（同分AP提高）且本机最终包内<=100ms才能生成交付；最终100ms来自规则，本机仍不能替代官方T4核验。模型+必要文件和ZIP都<=500MB。若不改善则不生成第五版CSV并如实报告。

资源预检：ConvNeXt尚未实测384训练容量，先做合成前向/反向/AdamW与推理延迟，不触碰结果选型。若容量或速度不能达标，记录停止原因，不偷偷改方案继续搜索。官方权重来源https://download.pytorch.org/models/convnext_tiny-983f1562.pth；文档https://docs.pytorch.org/vision/0.19/models/generated/torchvision.models.convnext_tiny.html；许可证沿用torchvision BSD全文，另归档convnext官方源文件和SHA。

实施与验证顺序：
1. 增加可审计内部划分、ConvNeXt二分类头与固定轮次保存；先加边界、无泄漏、恢复绑定和推理重载测试，再改实现。
2. 完整verify；归档权重来源、许可证、完整SHA；容量与真实路径测速。
3. 创建固定内部划分；顺序训练2个内部候选；单图比较4个候选并冻结选择报告。
4. 完整8轮最终重训，仅使用冻结轮次；固定配置验证对照第三版。
5. 达标后冻结测试预测、声明、包内测速、CRC/SHA/文件绑定和独立交付复核；用户自行上传。

全部文件位于E:\bisai\glaucoma_project；旧原始数据、前三版/第四版交付和模型保留。无新生产依赖、无自动平台提交。
