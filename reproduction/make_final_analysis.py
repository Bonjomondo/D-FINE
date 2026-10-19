"""Build the final reproducibility report and export figures from measured results."""
import csv
import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / 'reproduction'
OUT = REP / 'evaluation_audit'
result = json.loads((REP / 'self_trained_l_coco_metrics.json').read_text())
official = json.loads((REP / 'official_l_coco_metrics.json').read_text())
audit = json.loads((OUT / 'evaluation_audit.json').read_text())
assert audit['status'] == 'passed'
rows = [json.loads(line) for line in (ROOT / 'output/paper_l_coco_seed0/log.txt').read_text().splitlines()]
assert [r['epoch'] for r in rows] == list(range(74))
keys = ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large']
labels = ['AP', 'AP50', 'AP75', 'AP small', 'AP medium', 'AP large']
paper = [54., 71.6, 58.4, 36.5, 58., 71.9]
measured = [result['metrics_percent'][key] for key in keys]
delta = np.array(measured)-paper
with (OUT / 'paper_comparison.csv').open('w') as stream:
    writer = csv.writer(stream)
    writer.writerow(['metric', 'paper_rounded', 'self_trained', 'difference_AP_points', 'official_checkpoint_control'])
    for key, target, value in zip(keys, paper, measured):
        writer.writerow([key, target, value, value-target, official['metrics_percent'][key]])
with (OUT / 'training_curve.csv').open('w') as stream:
    writer = csv.writer(stream)
    writer.writerow(['epoch', *keys, 'train_loss'])
    for r in rows:
        writer.writerow([r['epoch']+1, *[v*100 for v in r['test_coco_eval_bbox'][:6]], r['train_loss']])
epochs = np.arange(1,75)
values = np.array([r['test_coco_eval_bbox'][:6] for r in rows])*100
plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                     'figure.facecolor':'white', 'axes.titleweight':'semibold'})
fig, axes = plt.subplots(1,2,figsize=(12.5,4.4), gridspec_kw={'width_ratios':[1.45,1]})
ax = axes[0]
ax.plot(epochs, values[:,0], color='#2463a6', linewidth=2, label='Self-trained validation AP')
ax.axhline(54.,color='#777777',linestyle='--',linewidth=1,label='Paper AP = 54.0')
ax.axvspan(72.5,74.5,color='#e9b44c',alpha=.3,label='Final 2 epochs: strong aug. off')
ax.scatter([74],[measured[0]],color='#2463a6',zorder=5)
ax.annotate(f'{measured[0]:.3f}',xy=(74,measured[0]),xytext=(54,48),
            arrowprops={'arrowstyle':'->','color':'#2463a6'},color='#2463a6',weight='bold')
ax.set(xlim=(1,75),ylim=(34,55.3),xlabel='Training epoch',ylabel='COCO AP (%)',title='D-FINE-L: full COCO training, seed 0')
ax.grid(axis='y',alpha=.2)
ax.legend(loc='lower right',frameon=False,fontsize=8.5)
ax = axes[1]
colors = ['#397c9d' if x>=0 else '#b97367' for x in delta]
ax.barh(labels,delta,color=colors,height=.62)
ax.axvline(0,color='#555555',linewidth=.8)
for y, value in enumerate(delta):
    ax.text(value+(0.015 if value>=0 else -.015),y,f'{value:+.3f}',
            va='center',ha='left' if value>=0 else 'right',fontsize=9)
ax.invert_yaxis()
ax.set(xlim=(-.37,.73),xlabel='Difference (AP points)',title='Difference to rounded paper values')
ax.grid(axis='x',alpha=.15)
fig.suptitle('COCO val2017: all 5,000 images, 640 x 640, EMA weights',fontsize=12,y=1.01)
fig.tight_layout()
fig.savefig(OUT/'reproduction_summary.png',dpi=180,bbox_inches='tight')
fig.savefig(OUT/'reproduction_summary.pdf',bbox_inches='tight')
plt.close(fig)
table = '\n'.join(f'| {label} | {target:.1f} | {value:.3f} | {value-target:+.3f} | {official["metrics_percent"][key]:.3f} |'
                  for key,label,target,value in zip(keys,labels,paper,measured))
first_best = max(rows[:72],key=lambda r:r['test_coco_eval_bbox'][0])
date = datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
report = f'''# D-FINE-L COCO 主实验：完成结果与论文评测核对

核对时间：{date}。用户选择的 COCO 主实验已完成；Objects365 保持延后，不自动执行。

## 结论

从 ImageNet 预训练 HGNetv2-B4 初始化，在完整 train2017 上执行 74 个 epoch 后，自训练最佳 EMA 权重在完整 val2017 上得到 **AP {measured[0]:.6f}**，相对论文 Table 1 的 54.0 差 **{delta[0]:+.6f} 个 AP 点**，按论文一位小数报告为 **54.0**。这次完成的是用户选定的 COCO-only L 模型主实验。

这是自训练结果，权重为 `output/paper_l_coco_seed0/best_stg2.pth`，SHA256 为 `{result['checkpoint_sha256']}`。单独下载的官方权重控制实验 AP 为 {official['metrics_percent']['AP']:.6f}；自训练相对官方权重仅差 {measured[0]-official['metrics_percent']['AP']:+.6f} 个 AP 点。

## 六项精度指标

以下差值单位均为 AP 点；论文值只有一位小数，不能把其视为未舍入的精确真值。

| 指标 | 论文 Table 1 | 自训练完整评测 | 自训练减论文 | 官方权重控制 |
|---|---:|---:|---:|---:|
{table}

中目标 AP 和 AP75 高于论文列出的舍入值，小目标和大目标 AP 略低，整体 AP 非常接近。该组结果说明主实验精度得到复现；这里只有 seed 0 一次训练，没有多种子方差、置信区间或显著性检验，不能据此宣称超过原论文方法。

## 评测方法逐项核对

| 项目 | 原文或官方实现 | 本次实际执行 |
|---|---|---|
| 比较对象 | Table 1 的 D-FINE-L，COCO-only，54.0 AP | 相同；不与 Objects365 57.1 AP 或 X 模型比较 |
| 划分 | Appendix A.1.2：train2017 / val2017 | 118,287 张训练图、5,000 张验证图，引用路径已完整检查 |
| 任务 | COCO bbox detection | bbox；80 类原始 category ID 映射 |
| 输入 | 官方验证 Resize 640×640 | 相同；直接 Resize，无翻转或多尺度 TTA |
| 权重 | 官方 Solver 验证优先使用 EMA | best_stg2.pth 内的 EMA，已核对加载日志 |
| 推理精度 | 官方 PyTorch evaluate 无 autocast | FP32；训练使用 AMP |
| 后处理 | 官方 sigmoid、跨查询类别 top-300、恢复原图坐标 | 相同，无额外 NMS 或置信度阈值调参 |
| AP | IoU 0.50:0.05:0.95，COCO 类别与召回阈值汇总 | 10 个 IoU 阈值、101 个 recall 点、80 类 |
| AP50 / AP75 | 固定 IoU=0.50 / 0.75 | 相同 |
| AP small/medium/large | COCO 原始 area 分组，32²、96² 分界 | 相同；依据原图标注面积，未用缩放后的面积替代 |
| 检测上限 | 标准 COCO maxDets=[1,10,100] | 相同；top-300 候选与 COCO maxDets 是不同处理阶段 |
| 评测程序 | 当前官方代码使用 faster-coco-eval | 1.8.0，并增加 pycocotools 2.0.11 独立复核 |
| 验证 batch | 官方配置 total_batch_size=64 | 本机 16；所有图像均参与，eval 模式，无 BatchNorm 训练统计更新 |

标准 COCO 实现的类别、IoU、recall、maxDets 和 area 参数均与加速实现逐项比较。对**同一次推理的全部 {audit['predictions']:,} 条预测**，两个后端前 12 项标准指标最大差为 **{audit['max_backend_difference_AP_points']:.12g} 个 AP 点**；独立复核相对训练后保存结果的最大差为 **{audit['max_stored_result_difference_AP_points']:.12g} 个 AP 点**。因此此次结论不依赖于用不同评测库“算高分”。

原始训练日志还打印自定义 Validator 的 precision/recall/F1/IoU；这些诊断输出与 COCO AP 采用不同汇总方式，论文 Table 1 的比较使用上述 CocoEvaluator 指标。本报告未将这些诊断量当作论文的 AP 或 AR。

## 训练协议与收敛

论文 Appendix A.1 / Table 6：HGNetv2-B4、AdamW、总 batch 32、base LR 2.5e-4、backbone LR 1.25e-5、weight decay 1.25e-4、EMA 0.9999、72 个增强 epoch + 2 个关闭强增强 epoch。本次保留这些设置及 FDR、GO-LSD 的官方实现。官方当前 YAML 默认 80 epoch 用于 EMA 搜索，本次独立配置固定为原文的 74。

训练日志包含连续的 epoch 索引 0–73，共 **74 个完整 epoch、273,504 个 batch 迭代**；训练耗时 **47 小时 32 分 04 秒**。10 月 19 日 11:12（北京时间）完成训练后独立完整评测。

| 完成 epoch | 验证 AP |
|---:|---:|
| 1 | {values[0,0]:.3f} |
| 20 | {values[19,0]:.3f} |
| 40 | {values[39,0]:.3f} |
| 60 | {values[59,0]:.3f} |
| 72 | {values[71,0]:.3f} |
| 73 | {values[72,0]:.3f} |
| 74 | {values[73,0]:.3f} |

AP 从初期逐步上升，后期约 53.4–53.7，关闭强增强后的最后两轮达到 53.827 和 53.951。最后两轮训练 loss 的下降同时受到增强变化的影响，应与验证 AP 一起阅读。

**官方阶段切换的一个细节**：第 73 轮开始前，源码按设计回载阶段一最佳 `best_stg1.pth`；本次阶段一最佳位于第 {first_best['epoch']+1} 轮，AP {first_best['test_coco_eval_bbox'][0]*100:.6f}。因此最终 checkpoint 的内部 `last_epoch=65`、EMA updates=243936，是从阶段一最佳状态继续两轮后的计数；74 个训练循环确实都已执行。应结合完整训练日志理解这一回退计数。

## 硬件、实现差异与效率指标

本机单张 RTX 5090 D v2，Python 3.11.16、PyTorch 2.14.0+cu130；论文报告训练使用四张 RTX 4090。为了保持总 batch 32，本次使用逐块激活重计算，并在重计算时保留原始 BatchNorm 缓冲区。数值对照记录在 `checkpoint_equivalence.json`：预测与损失通过严格容差检查、所有归一化缓冲区逐元素一致、梯度差异处于原始 CUDA 重复反向传播误差范围内。官方模型与训练源码文件未修改。不同硬件、库版本、随机数流与归约顺序仍可能带来小幅数值差异。

模型统计为 **30,724,195 个部署参数、91.0344 GFLOPs（640 输入）**，对应论文舍入的 **31M / 91 GFLOPs**。

论文的 **8.07 ms** 是 **NVIDIA T4 + TensorRT FP16** 的端到端延迟。本次没有在 T4/TensorRT 上复现这一速度指标。PyTorch 验证的批量吞吐、整段评测时间或训练步耗时都不能直接作为该延迟的对照。

## 文件与可复查证据

- 最终权重：`output/paper_l_coco_seed0/best_stg2.pth`
- 自训练完整评测：`reproduction/self_trained_l_coco_metrics.json`
- 原始评测输出：`reproduction/logs/self_trained_l_coco_eval.log`
- 双后端独立复核：`reproduction/evaluation_audit/evaluation_audit.json`
- 全部预测（gzip JSON）：`reproduction/evaluation_audit/self_trained_predictions.json.gz`
- 80 类 AP：`reproduction/evaluation_audit/per_category_ap.csv`
- 论文差值表：`reproduction/evaluation_audit/paper_comparison.csv`
- 74 轮训练曲线数据与图：`training_curve.csv`、`reproduction_summary.png`、`reproduction_summary.pdf`（均在本目录）

原文证据：本地 `reproduction/paper/paper.pdf` 的 §5.1、Table 1、Appendix A.1 与 Table 6。
官方 COCO API 定义：https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py 。
官方代码：https://github.com/Peterande/D-FINE ，固定提交 `956d1709314c2c6a4df6f34de232054578a7449f`。
'''
(OUT/'REPORT_CN.md').write_text(report)
print(OUT/'REPORT_CN.md')
