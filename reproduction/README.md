# D-FINE 原文复现记录

本目录保存论文复现配置、下载器、运行入口和实测结果。官方仓库：
https://github.com/Peterande/D-FINE ，固定提交为
`956d1709314c2c6a4df6f34de232054578a7449f`。
论文以工作区提供的 PDF 为准，副本与文本位于 `paper/`；原文链接：
https://arxiv.org/abs/2410.13842 。官方模型和训练源码保持原样，新增实现均在本目录。

**用户已选择先完成 D-FINE-L 的 COCO 主实验：约 20.4 GB 下载数据、batch 32、74 epoch、论文 AP 54.0 对照。** 范围保存在 `scope.json`。Objects365、其他模型和消融不属于当前自动执行范围，额外下载和预训练队列保持停止。

**2026-10-19 已完成：自训练权重完整 val2017 AP 53.951238，按论文一位小数为 54.0。** 官方 pycocotools 与 faster-coco-eval 对同一组 150 万条预测的 12 项指标完全一致。完整逐项核对、训练曲线和差值分析见 [最终分析报告](evaluation_audit/REPORT_CN.md)。

## 已完成与当前进度

- 项目内 `.venv` 已安装官方依赖，复用 `UAV_CV` 的 Python 3.11.16、PyTorch 2.14.0+cu130、torchvision 0.29.0+cu130；`environment.lock.txt` 记录版本。未修改已有环境。
- 已下载 ImageNet 初始化的 `weight/hgnetv2/PPHGNetV2_B4_stage1.pth` 和作者的 `weight/official/dfine_l_coco.pth`。
- 完整 COCO2017 标注、5,000 张验证图像和 118,287 张训练图像已下载并通过 CRC、图像数量及引用路径检查。
- 作者 L 模型权重在完整 val2017 上实测 AP **53.980919**，与原文 **54.0** 的舍入结果一致。**这是官方权重评测，并非本机从头训练的结果。** 原始输出在 `logs/official_l_coco_eval.log`，机器可读结果在 `official_l_coco_metrics.json`。
- 原文 batch 32、最大多尺度输入 800 的数值对照、显存检查、真实数据训练引擎检查及官方 Solver 启动检查已通过。
- 74 个完整 epoch 已全部执行，训练耗时 47 小时 32 分 04 秒，2026-10-19 11:12 完成独立完整评测。自训练结果在 `self_trained_l_coco_metrics.json`；最佳权重在 `output/paper_l_coco_seed0/best_stg2.pth`。当前无训练进程运行。

实时状态以 `status.json` 和以下命令为准，以上记录不会自动更新：

```bash
cd /home/ubuntu/文档/ZHJ_UAV/D-FINE
.venv/bin/python reproduction/status.py
tmux attach -t dfine-reproduction
tail -f reproduction/logs/download_coco.log
tail -f reproduction/logs/train_l_coco.log
```

## 已完成的完整验证集评测

| 指标 | 原文 L COCO | 自训练完整评测 | 官方权重控制 |
|---|---:|---:|---:|
| AP | 54.0 | 53.951 | 53.981 |
| AP50 | 71.6 | 71.677 | 71.640 |
| AP75 | 58.4 | 58.744 | 58.381 |
| AP small | 36.5 | 36.363 | 36.468 |
| AP medium | 58.0 | 58.525 | 58.026 |
| AP large | 71.9 | 71.691 | 71.895 |

评测使用官方 `evaluate()`、原图尺寸恢复、80 类映射、640×640 输入和全部 5,000 张图片。验证 batch 为 16，仅调整评测内存使用。未用子集 AP 代替完整验证集结果。

## 原文训练配置

以论文附录 A.1 / Table 6 为依据：总 batch 32、AdamW、EMA 0.9999、AMP、seed 0，L 的 base LR 为 2.5e-4、backbone LR 为 1.25e-5、weight decay 为 1.25e-4，保留官方 FDR、GO-LSD、损失权重与多尺度增强。

当前官方 L/X 配置是 80 epoch，M/S 是 132 epoch，注释说明用于 EMA 搜索。本目录用单独的 YAML 覆盖为论文日程，未覆盖官方 YAML：

| 模型 | COCO 原文日程 | Objects365 预训练 | COCO 微调 |
|---|---:|---:|---:|
| S | 120 + 4 = 124 | 28 | 58 |
| M | 120 + 4 = 124 | 29 | 49 |
| L | 72 + 2 = 74 | 21 | 32 |
| X | 72 + 2 = 74 | 21 | 31 |

配置均已生成在 `configs/`。**自动队列仅执行用户选定的 L COCO 主实验。Objects365 额外下载与等待队列已停止，用户已明确选择先完成 COCO。** 其他模型、消融实验及 T4 TensorRT 延迟实验不属于当前执行范围。S/M/X 的初始化权重也尚未下载。本次任务的完成标准是当前选定主实验及其完整评测。

## 单卡适配和数值验证

本机是单张 RTX 5090 D v2，显存约 24 GB；原文实验描述使用四张 RTX 4090，延迟使用 T4。原始代码 batch 32 在 800 输入时超过本机显存，见 `memory_l_b32_s800.json`。

当前采用 **逐块激活重计算**（`activation_checkpoint.py`），保持完整 batch 32，不拆分 batch，不累积梯度，不缩小输入。它重计算 HG、编码器与解码器块，在重计算阶段临时使用 BatchNorm 缓冲区副本，保证每步运行统计只更新一次。模型结构、参数键名、优化器和训练日程均不变。`run.py` 在进程内包装官方类，官方源文件未修改。

`checkpoint_equivalence.json` 检查了预测、损失、569 个梯度张量及所有模型缓冲区。预测/损失通过严格容差检查，归一化缓冲区逐元素完全一致。CUDA `grid_sampler_2d_backward_cuda` 没有确定性实现，原始代码的重复反向传播也有误差；重计算梯度误差落在实测重复噪声范围内。记录包含重复对照和验收标准，未宣称逐位相同的训练结果。

真实 COCO 数据检查在 800×800、batch 32 下执行 2 步，共 **2.92 秒**，平均 **1.46 秒/步**，峰值显存约 **10.26 GiB**，见 `checkpoint_real_batch_smoke.json`。原生 AMP 首步调整 loss scale 后，下一步实际完成有限梯度的优化器更新。另一次冷启动最大尺度合成显存检查耗时 2.13 秒。这些检查为了在训练集就绪前核验引擎，真实数据检查使用验证图像与训练增强，结果单独标记，不计入正式训练。正式流水线只用 train2017 训练。

74 epoch × floor(118287/32) = **273,504 个 batch 迭代**。最终实测训练耗时 **47 小时 32 分 04 秒**，末轮平均 0.5964 秒/迭代；早期 5–7 天估算基于冷启动最大输入检查，实际完整训练已完成。

Python/PyTorch/CUDA 与论文时代的环境不同，以支持 RTX 5090；单卡与四卡的随机数序列和浮点归约也可能不同，因此同一训练协议并不保证逐位相同或恰好同一 AP。

## 数据与自动执行

数据位置：`../datasets/coco2017/`。来源为 COCO 官方 `images.cocodataset.org`，使用官方可用的 HTTP 下载端点；未关闭 HTTPS 证书验证。归档以解压 CRC 校验、文件数和 SHA256 记录在数据目录的 `*.verified.json` 中。

`prepare_coco.py` 下载完整标准 split。`verify_data.py` 在开训前检查 split 图像数、80 类及每张标注引用图像的存在性。训练图像未缩放重编码，未替换为 UAV 或其他数据。

`pipeline.py` 自动执行：官方权重完整评测 → 等待完整训练归档 → 数据检查 → ImageNet backbone 初始化的 L 模型训练 → 自训练 best checkpoint 完整评测。训练不使用官方 COCO 检测权重作为初始化。

正式训练输出在 `output/paper_l_coco_seed0/`；官方权重评测输出在 `output/official_l_coco_eval/`。流水线失败会将错误写入 `status.json`；重启时若已有 `last.pth`，会按官方恢复机制续训。

查看持久会话：

```bash
tmux list-windows -t dfine-reproduction
```

仅在相应任务已停止时重启，避免重复启动：

```bash
cd /home/ubuntu/文档/ZHJ_UAV/D-FINE
tmux new-window -t dfine-reproduction -n data 'cd /home/ubuntu/文档/ZHJ_UAV && python -u D-FINE/reproduction/prepare_coco.py > D-FINE/reproduction/logs/download_coco.log 2>&1'
tmux new-window -t dfine-reproduction -n pipeline 'cd /home/ubuntu/文档/ZHJ_UAV/D-FINE && .venv/bin/python -u reproduction/pipeline.py >> reproduction/logs/pipeline.log 2>&1'
```

流水线设置 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`，用于新 PyTorch 读取作者发布及本机保存的完整训练状态；该入口只应加载可信 checkpoint。

将来有四卡环境时，可使用官方训练入口和本目录论文配置：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 train.py -c reproduction/configs/paper_l_coco.yml --use-amp --seed=0
```

移动项目后先运行 `make_configs.py` 重新生成本机绝对数据路径。

## 尚待完成的 Objects365 分支

**此分支是额外的预训练实验，复现 COCO-only 的 54.0 AP 不需要下载 Objects365。** 完整 Objects365 分支需要额外约 369.8 GB 压缩数据；加 COCO 后总网络下载量约 390.2 GB。

用户提供的公开镜像 https://huggingface.co/datasets/ivanroubicek/objects365 已核验可匿名下载，固定版本为 `45d797979309783ad1730263fa06cbab37ebbc80`。该版本缺 `patch47.tar.gz`，没有验证集。已通过 Ultralytics 官方下载脚本中的原始直链补齐来源，97 个文件（51 个训练分包、44 个验证分包、两份标注）全部通过匿名 HEAD 检查。完整大小、来源、镜像 SHA256 在 `objects365/download_manifest.json`，无需注册智源账号。之前把注册当成下载前提的判断不完整，已修正。

`prepare_objects365.py` 已完成两份标注及部分验证图片的下载/解压，按镜像 SHA256 或原始 MD5、归档 CRC 核验，并逐包解压后回收本任务压缩包。用户对额外数据量提出疑问后，**下载进程和 Objects365 后续等待队列均已停止**，保留约 7.2 GiB 可续用数据，没有下载完整的 370 GB。状态见 `objects365/download_status.json`、`objects365/pipeline_status.json`。COCO 训练继续运行。

数据就绪后，按原文合并训练集与验证集中除前 5,000 张以外的图片，并预先将长边超过 640 的图片缩到 640。官方实际脚本位置为 `tools/dataset/remap_obj365.py` 和 `tools/dataset/resize_obj365.py`（README 的旧路径少了 `dataset/`）。原文 L 为 21 epoch 预训练 + 32 epoch 微调，目标 AP 57.1；当前 README 的 E25 权重属于后来的 25 epoch 版本，不可替代这一原文实验。

`preprocess_objects365.py` 用流式 JSON 和 SQLite 执行上述处理，避免把全部大规模标注同时载入内存。`objects365/preprocessing_equivalence.json` 记录与作者 remap/resize 脚本的测试对照：生成 JSON 语义及缩放图片字节一致，包括原始缺失图片的处理。

`objects365_pipeline.py` 已准备好在数据完整后预处理，并在 COCO 实验完成后，依次执行 L 的 21 epoch 预训练与 32 epoch COCO 微调。**用户已将此分支延后；队列当前未运行，也不会在 COCO 完成后自行恢复。** 下载器及队列会检查 `scope.json`，当前范围下直接退出。只有用户之后明确要求包含此分支，才调整范围并恢复。`configs/paper_*_obj365.yml` 与 `configs/paper_*_obj2coco.yml` 已准备，对应完整训练尚未执行。
