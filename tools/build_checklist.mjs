import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const option=(flag,fallback)=>{const index=process.argv.indexOf(flag);if(index<0)return fallback;if(!process.argv[index+1])throw new Error(`Missing ${flag}`);return process.argv[index+1];};
const optimized = process.argv.includes('--optimized');
const output=path.resolve(root,option('--output',optimized?'output/optimized':'output'));
if(!output.startsWith(root+path.sep))throw new Error('Output must remain inside the project');
const provenance = JSON.parse(await fs.readFile(path.resolve(root,option('--provenance','reports/pretrain_provenance.json')),'utf8'));
const weight = provenance.find(x=>x.path.endsWith('.pth'));
const license = await fs.readFile(path.join(root,'licenses/torchvision-BSD.txt'),'utf8');
const rows = [
 ['序号','名称','声明内容','依据或材料'],
 ['A1','所用开源模型','ResNet-18；torchvision 官方 IMAGENET1K_V1 权重','https://docs.pytorch.org/vision/0.19/models/generated/torchvision.models.resnet18.html'],
 ['A2','权重公开链接',weight.url,'公网直接下载，无需登录'],
 ['A3','许可证全文','BSD 3-Clause，原文见“许可证”工作表及随包 licenses/torchvision-BSD.txt','https://github.com/pytorch/vision/blob/v0.19.1/LICENSE'],
 ['A4','上游训练数据','ImageNet-1K 通用自然图像；非医学预训练','https://github.com/pytorch/vision/blob/v0.19.1/torchvision/models/resnet.py'],
 ['A5','预训练权重 SHA-256',weight.sha256,'对下载权重计算完整 SHA-256，匹配官方文件名哈希前缀'],
 ['A6','推理加速精度','FP16 autocast前向；FP32 sigmoid；Python浮点等权融合；参数FP32；eval、无梯度','predict.py / cost.py 共享 Predictor'],
 ['B1','训练数据使用','仅官方 train；不使用外部医学数据','train.py；data.py'],
 ['B2','验证集使用','仅用于模型选择；未并入训练','AUROC 优先，同时报告 average precision 和梯形 AUPRC'],
 ['B3','测试集使用','仅固定模型最终推理及提交运行检查；不调参、不训练、不拟合统计量','逐图确定性预处理，无跨测试样本统计'],
 ['B4','预处理','RGB；逐图黑边裁剪阈值12；等比例填充224×224；固定ImageNet均值和标准差','data.py'],
 ['B5','训练增强','水平/垂直翻转、±15°旋转、温和颜色增强；仅训练阶段','推理关闭随机增强'],
 ['B6','TTA与模型集成','均未使用','单个ResNet-18'],
 ['B7','类别阈值',0.5,'Probability≥0.5 时 Pred_Label=1'],
 ['C1','本地训练环境','Python3.9.13；torch2.4.1+cu124；torchvision0.19.1+cu124；RTX4060 Laptop 8GB','requirements-local.lock.txt；训练环境记录'],
 ['C2','官方环境复测','未验证','官方目标：Python3.12，torch2.10.0，torchvision0.25.0，T4 16GB'],
 ['C3','官方100ms测速','未验证：尚无官方100张测试图和5张预热图','cost.py：batch=1、3轮，含读图至写出结果'],
 ['C4','模型体积','以最终打包文件清单中的实测字节数为准','运行时权重及必要配套文件合计≤500MB'],
];
if (optimized) {
 const selection=JSON.parse(await fs.readFile(path.join(output,'selection_report.json'),'utf8'));
 const configurations=await Promise.all(selection.models.map(async x=>JSON.parse(await fs.readFile(path.join(path.dirname(x.checkpoint),'config.json'),'utf8'))));
 const architectures=[...new Set(configurations.map(x=>x.architecture||'resnet18'))];
 const selectedWeights=provenance.filter(x=>x.path.endsWith('.pth')&&architectures.some(architecture=>path.basename(x.path).startsWith(architecture+'-')));
 if(selectedWeights.length!==architectures.length)throw new Error('Missing selected model provenance');
 const architectureNames={resnet18:'ResNet-18 / IMAGENET1K_V1',resnet50:'ResNet-50 / IMAGENET1K_V2',convnext_tiny:'ConvNeXt-Tiny / IMAGENET1K_V1',efficientnet_v2_s:'EfficientNetV2-S / IMAGENET1K_V1'};
 rows.find(x=>x[0]==='A1')[2]=architectures.map(x=>{if(!architectureNames[x])throw new Error('Unknown declaration architecture');return architectureNames[x];}).join('；');
 rows.find(x=>x[0]==='A1')[3]=architectures.map(x=>`https://docs.pytorch.org/vision/0.19/models/generated/torchvision.models.${x}.html`).join('\n');
 rows.find(x=>x[0]==='A2')[2]=selectedWeights.map(x=>x.url).join('\n');
 rows.find(x=>x[0]==='A4')[3]=architectures.map(x=>`https://github.com/pytorch/vision/blob/v0.19.1/torchvision/models/${x==='convnext_tiny'?'convnext':x==='efficientnet_v2_s'?'efficientnet':'resnet'}.py`).join('\n');
 rows.find(x=>x[0]==='A5')[2]=selectedWeights.map(x=>`${path.basename(x.path)}\n${x.sha256}`).join('\n');
 const descriptions=selection.models.map(x=>`${path.basename(path.dirname(x.checkpoint))}: 水平翻转TTA=${x.tta}`);
 rows.find(x=>x[0]==='B4')[2]='RGB；逐图黑边裁剪阈值12；按入选权重所记载分辨率等比例填充；固定ImageNet归一化';
 rows.find(x=>x[0]==='B6')[2]=`${selection.models.length}个模型等权概率平均；${descriptions.join('；')}`;
 rows.find(x=>x[0]==='B6')[3]='model/config.json；selection_report.json';
 rows.find(x=>x[0]==='C3')[3]=`本机验证集100张×3轮平均${selection.local_mean_ms.toFixed(2)}ms；不能替代官方T4测评`;
 if(configurations.every(x=>x.checkpoint_epoch>0)) {
  rows.find(x=>x[0]==='B2')[2]='官方train内部留出选架构/轮次/TTA；官方valid作逐轮监控及最终提升检查，不选轮次、不更新参数';
  rows.find(x=>x[0]==='B2')[3]='internal_selection.json；固定checkpoint_epoch，AUROC优先、AP同分辅助';
 } else if(configurations.some(x=>x.checkpoint_epoch>0)) {
  rows.find(x=>x[0]==='B2')[2]='R50沿用官方valid选型；ConvNeXt用train内部留出选架构/轮次/TTA；组合在官方valid复核';
  rows.find(x=>x[0]==='B2')[3]='internal_selection.json；selection_report.json；官方valid未并入训练，已多次用于探索性选型';
 }
 if(configurations.some(x=>x.augmentation==='color_robust')) {
  rows.find(x=>x[0]==='B5')[2]='color_robust成员：H/V翻转、±15°旋转；颜色抖动(亮/对/饱/色=.30/.30/.25/.04)，灰度p=.10，3×3模糊σ=.1–.8、p=.10；其余成员沿用原训练配方。均仅训练。';
  rows.find(x=>x[0]==='B5')[3]='run/config.json augmentation；data.py；OPTIMIZATION-v8.md';
  rows.find(x=>x[0]==='B2')[3]='原图/固定颜色扰动内部比较；固定checkpoint_epoch；validation_ci_gate.json仅描述性工程筛选；官方valid未并入训练';
 }
}
const wb=Workbook.create();
const sheet=wb.worksheets.add('模型与数据声明');
sheet.showGridLines=false;
sheet.getRange(`A1:D${rows.length}`).values=rows;
sheet.getRange(`A1:D${rows.length}`).format.font={name:'Microsoft YaHei',size:11};
sheet.getRange(`A1:D${rows.length}`).format.wrapText=true;
sheet.getRange(`A1:D${rows.length}`).format.verticalAlignment='center';
sheet.getRange('A:A').format.columnWidth=7;
sheet.getRange('B:B').format.columnWidth=22;
sheet.getRange('C:C').format.columnWidth=67;
sheet.getRange('D:D').format.columnWidth=63;
sheet.getRange(`A2:D${rows.length}`).format.rowHeight=62;
sheet.getRange('A6:D6').format.rowHeight=100;
if(optimized&&rows.find(x=>x[0]==='B5')[2].includes('color_robust'))sheet.getRange('A12:D12').format.rowHeight=110;
sheet.getRange('A1:D1').format={fill:'#24476C',font:{name:'Microsoft YaHei',size:11,bold:true,color:'#FFFFFF'},rowHeight:28};
sheet.getRange('C14').setNumberFormat('0.0');
sheet.freezePanes.freezeRows(1);
const lic=wb.worksheets.add('许可证');
lic.showGridLines=false;
lic.getRange('A1').values=[['torchvision BSD 3-Clause License']];
lic.getRange('A1').format.font={name:'Arial',size:14,bold:true};
const lines=license.split('\n');
lic.getRange(`A3:A${lines.length+2}`).values=lines.map(x=>[x]);
lic.getRange(`A3:A${lines.length+2}`).format.font={name:'Arial',size:11};
lic.getRange('A:A').format.columnWidth=110;
lic.getRange(`A3:A${lines.length+2}`).format.rowHeight=20;
wb.recalculate();
const result=await wb.inspect({kind:'table',range:'模型与数据声明!A1:D7',include:'values',tableMaxRows:7,tableMaxCols:4,maxChars:1600});
console.log(result.ndjson);
await fs.mkdir(output,{recursive:true});
await (await SpreadsheetFile.exportXlsx(wb)).save(path.join(output,'模型核验材料与数据申明清单.xlsx'));
const previews=[['模型与数据声明','A1:D7','checklist-preview.png'],['许可证','A1:A32','license-preview.png']];
if(optimized) previews.push(['模型与数据声明',`A8:D${rows.length}`,'optimized-settings-preview.png']);
for(const [name,range,file] of previews){
 const preview=await wb.render({sheetName:name,range,scale:1,format:'png'});
 await fs.writeFile(path.join(root,'tmp',file),new Uint8Array(await preview.arrayBuffer()));
}
console.log('Workbook saved');
