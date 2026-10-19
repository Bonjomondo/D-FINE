# D-FINE-L COCO 主实验：完成结果与论文评测核对

核对时间：2026-10-19T15:29:16.884113+08:00。用户选择的 COCO 主实验已完成；Objects365 保持延后，不自动执行。

## 结论

从 ImageNet 预训练 HGNetv2-B4 初始化，在完整 train2017 上执行 74 个 epoch 后，自训练最佳 EMA 权重在完整 val2017 上得到 **AP 53.951238**，相对论文 Table 1 的 54.0 差 **-0.048762 个 AP 点**，按论文一位小数报告为 **54.0**。这次完成的是用户选定的 COCO-only L 模型主实验。

这是自训练结果，权重为 `output/paper_l_coco_seed0/best_stg2.pth`，SHA256 为 `e93b13325d84dcb1ddf9a8a501a278b06c7970803e2c771595783e91446a5d9c`。单独下载的官方权重控制实验 AP 为 53.980919；自训练相对官方权重仅差 -0.029681 个 AP 点。

## 六项精度指标

以下差值单位均为 AP 点；论文值只有一位小数，不能把其视为未舍入的精确真值。

| 指标 | 论文 Table 1 | 自训练完整评测 | 自训练减论文 | 官方权重控制 |
|---|---:|---:|---:|---:|
| AP | 54.0 | 53.951 | -0.049 | 53.981 |
| AP50 | 71.6 | 71.677 | +0.077 | 71.640 |
| AP75 | 58.4 | 58.744 | +0.344 | 58.381 |
| AP small | 36.5 | 36.363 | -0.137 | 36.468 |
| AP medium | 58.0 | 58.525 | +0.525 | 58.026 |
| AP large | 71.9 | 71.691 | -0.209 | 71.895 |

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

标准 COCO 实现的类别、IoU、recall、maxDets 和 area 参数均与加速实现逐项比较。对**同一次推理的全部 1,500,000 条预测**，两个后端前 12 项标准指标最大差为 **0 个 AP 点**；独立复核相对训练后保存结果的最大差为 **1.11022302463e-14 个 AP 点**。因此此次结论不依赖于用不同评测库“算高分”。

原始训练日志还打印自定义 Validator 的 precision/recall/F1/IoU；这些诊断输出与 COCO AP 采用不同汇总方式，论文 Table 1 的比较使用上述 CocoEvaluator 指标。本报告未将这些诊断量当作论文的 AP 或 AR。

## 训练协议与收敛

论文 Appendix A.1 / Table 6：HGNetv2-B4、AdamW、总 batch 32、base LR 2.5e-4、backbone LR 1.25e-5、weight decay 1.25e-4、EMA 0.9999、72 个增强 epoch + 2 个关闭强增强 epoch。本次保留这些设置及 FDR、GO-LSD 的官方实现。官方当前 YAML 默认 80 epoch 用于 EMA 搜索，本次独立配置固定为原文的 74。

训练日志包含连续的 epoch 索引 0–73，共 **74 个完整 epoch、273,504 个 batch 迭代**；训练耗时 **47 小时 32 分 04 秒**。10 月 19 日 11:12（北京时间）完成训练后独立完整评测。

| 完成 epoch | 验证 AP |
|---:|---:|
| 1 | 35.600 |
| 20 | 51.383 |
| 40 | 53.020 |
| 60 | 53.553 |
| 72 | 53.468 |
| 73 | 53.827 |
| 74 | 53.951 |

AP 从初期逐步上升，后期约 53.4–53.7，关闭强增强后的最后两轮达到 53.827 和 53.951。最后两轮训练 loss 的下降同时受到增强变化的影响，应与验证 AP 一起阅读。

**官方阶段切换的一个细节**：第 73 轮开始前，源码按设计回载阶段一最佳 `best_stg1.pth`；本次阶段一最佳位于第 64 轮，AP 53.656871。因此最终 checkpoint 的内部 `last_epoch=65`、EMA updates=243936，是从阶段一最佳状态继续两轮后的计数；74 个训练循环确实都已执行。应结合完整训练日志理解这一回退计数。

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
