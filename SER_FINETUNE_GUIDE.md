# SER 微调指南

本文只负责 **SER 协议、命令语义和指标口径**。环境安装与版本见 [ENVIRONMENT.md](./ENVIRONMENT.md)；服务器连接、代码同步、smoke 和排障见 [GPU-RUNBOOK.md](./GPU-RUNBOOK.md)。

## 1. 协议约定

| 数据集 | 类别数 | fold | 默认长度 | nested 选择上限 |
|---|---:|---|---:|---:|
| EMO-DB | 7 | 1–10 | 640 | 80 epochs |
| IEMOCAP-5531 | 4 | 1–5 | 1024 | 20 epochs |
| RAVDESS-Speech | 8 | 0–4 | 512 | 80 epochs |
| RAVDESS-Song | 6 | 0–4 | 640 | 80 epochs |

- `nested`：用 `inner_train/inner_valid` 选 epoch，再从 EAT 预训练权重重新初始化，在完整 `train` 上训练所选 epoch，最后只评估一次 `valid`。
- `fixed`：只读取 `train/valid`，默认完整训练 60 epochs，不用 `valid` 选 checkpoint，最后只评估一次。
- 两种协议均保留 EAT AudioSet 归一化：`mean=-4.268`、`std=4.569`。

## 2. 路径与环境

```bash
export FAIRSEQ_ROOT=/root/songhao/ProJ/fairseq
export SER_DATA_ROOT=/root/songhao/datasets/SER_DATASET
export SER_OUTPUT_ROOT=/root/songhao/experiments/EAT2EMO
export SER_PRETRAINED=/root/songhao/ProJ/resources/EAT_Weight/EAT-base_epoch30_pt.pt
export PYTHON_BIN=/root/miniconda3/envs/eat2emo/bin/python
export PYTHONDONTWRITEBYTECODE=1
```

当前服务器环境为 `eat2emo`（见 [ENVIRONMENT.md](./ENVIRONMENT.md)）。SER 流程不使用 `CHECKPOINT_ROOT`；该变量只被旧的非 SER 脚本使用。

默认预训练权重为 `/root/songhao/ProJ/resources/EAT_Weight/EAT-base_epoch30_pt.pt`，可用 `SER_PRETRAINED` 覆盖。

> 本仓库 `datasets/` 下的本地数据和 `datasets/manifests/*` 是旧 `train/eval/test` 快照，不是当前 nested/fixed 协议的输入；正式流程只使用服务器 `SER_DATASET`。
>
> 训练默认启用 **fbank 特征缓存**：首次 run 在 `${SER_DATA_ROOT}/feature_cache/<dataset>/...` 写入 `.npy`，后续 epoch/fold/seed 直接复用；`roll_aug/noise` 训练会自动绕过缓存。缓存会把一次 fbank 的数值结果固定下来（与实时计算差异约 1e-7），正式对比时不要混用 cached / non-cached 结果；删除该目录可重建。

## 3. 生成并审计 manifests

```bash
cd /root/songhao/ProJ/EAT2EMO
$PYTHON_BIN tools/prepare_ser_manifests.py \
  --datasets-root /root/songhao/datasets/SER_DATASET \
  --manifest-root /root/songhao/datasets/SER_DATASET \
  --output /root/songhao/datasets/SER_DATASET/manifests
```

每个 `fold_XX` 包含：

- `train/valid/inner_train/inner_valid.tsv`
- 对应的 `.lbl`
- `label_descriptors.csv`
- 带样本数、分组、音频格式和文件哈希审计的 `split.json`

生成器会强制检查总数 `535/5531/1440/1012`、说话人零交叉、outer-valid 唯一覆盖，以及所有 WAV 为单声道 16 kHz。任一检查失败都会终止。

IEMOCAP 如果要从 parquet 重建，先用下面的工具生成 `metadata.csv` 和 `SessionN/` 布局（默认要求 5531 条 4 类样本；非协议类别会被跳过）：

```bash
$PYTHON_BIN tools/extract_iemocap_parquet.py /path/to/parquet \
  --output /root/songhao/datasets/SER_DATASET/IEMOCAP-5531
```

输出可直接被 `prepare_ser_manifests.py` 读取；如果输入不是 5531 条子集，工具会在写文件前报错，避免生成半成品。

## 4. 单 fold 训练

Nested 完整流程：

```bash
bash scripts/finetuning_SER.sh \
  --dataset emodb --fold 1 --seed 3949 --protocol nested
```

也可分阶段执行：

```bash
bash scripts/finetuning_SER.sh \
  --dataset emodb --fold 1 --seed 3949 --protocol nested --stage select
bash scripts/finetuning_SER.sh \
  --dataset emodb --fold 1 --seed 3949 --protocol nested --stage final
```

Fixed：

```bash
bash scripts/finetuning_SER.sh \
  --dataset emodb --fold 1 --seed 3949 --protocol fixed --fixed-epochs 60
```

输出按以下目录隔离，已存在时默认拒绝覆盖：

```text
${SER_OUTPUT_ROOT}/{dataset}/{protocol}/{config}/seed_N/fold_XX/{selection|final|fixed}
```

## 5. 服务器 smoke test

先使用独立输出目录，对每种协议实际跑 1 epoch：

```bash
export SER_OUTPUT_ROOT=/root/songhao/experiments/EAT2EMO_smoke

bash scripts/finetuning_SER.sh \
  --dataset emodb --fold 1 --seed 3949 --protocol nested \
  --selection-max-epochs 1

bash scripts/finetuning_SER.sh \
  --dataset emodb --fold 1 --seed 3949 --protocol fixed \
  --fixed-epochs 1
```

`--dry-run` 只校验输入并打印 Fairseq 命令，不会训练或写入实验目录。

## 6. 批量实验与汇总

默认运行四个数据集、seeds `3949,42,2026,1024,666`：

```bash
export SER_OUTPUT_ROOT=/root/songhao/experiments/EAT2EMO
bash scripts/run_all_SER.sh --protocol nested
bash scripts/run_all_SER.sh --protocol fixed --fixed-epochs 60
```

仅运行指定数据集、seeds 或 folds：

```bash
bash scripts/run_all_SER.sh \
  --datasets emodb,ravdess-song \
  --seeds 3949,42 \
  --folds 1,2 \
  --protocol nested
```

- `--seeds all`：全部 5 个参考 seed（默认）；`--seeds 3949`：只跑一个。
- `--folds all`：每个数据集自己的 full range；`--folds 1` 或 `--folds 1,2`：只跑指定折。
- 显式 fold 必须对每个选中的 dataset 合法（EMO-DB 1–10、IEMOCAP 1–5、RAVDESS 0–4），否则在启动前报错。

只有 `--folds all` 时才会在数据集完成后自动严格汇总；显式 fold 子集会跳过自动 summary，per-fold 的 `metrics.json` 仍保留。也可手动执行：

```bash
$PYTHON_BIN evaluation/summarize_ser.py \
  /root/songhao/experiments/EAT2EMO/emodb/nested/eat_base_sr16k_audiosetnorm_bs32 \
  --dataset emodb --protocol nested \
  --config eat_base_sr16k_audiosetnorm_bs32
```

每个 fold 保存 `metrics.json`、`predictions.npz`、`paths.txt`、`confusion_matrix.csv`和 `eval.log`。汇总产生 JSON 主结果与 fold CSV：

- EMO-DB 主结果：每个 seed 池化 10 折、535 条 OOF 预测，再对 5 seeds 计算均值±标准差。
- 其他数据集：先在每个 seed 内汇总 folds，再计算 5 seeds 的均值±标准差。
- 逐 fold 的 Accuracy/WA、UAR、Macro-F1、loss 及 fold mean±std 始终保留为补充。

## 7. 运行原则

- 调度器按训练样本数计算 `ceil(samples/32) × epochs` 个 updates，warm-up 为总 updates 的 10%，保留 cosine 与 `min_lr=1e-6`。
- 音频解码失败、空音频或无效样本立即终止，不跳过、不用其他样本补位。
- `fixed` 与 nested `final` 训练期间禁用验证，只在结束后评估 outer-valid 一次。
- 单 fold 缺类时，UAR/Macro-F1 只对该 fold 实际出现的类计算；EMO-DB OOF 主指标使用完整 7 类。
