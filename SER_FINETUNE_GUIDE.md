# SER 微调指南

本文负责 **SER 协议、命令语义和指标口径**。环境安装与版本见 [ENVIRONMENT.md](./ENVIRONMENT.md)；服务器连接、代码同步、smoke 和排障见 [GPU-RUNBOOK.md](./GPU-RUNBOOK.md)。

## 1. 协议：earlystop

所有数据集统一使用固定 outer split + inner-valid early stopping：

```text
outer-train
├── inner_train   -> 训练
└── inner_valid   -> 每个 epoch 评估，选 best checkpoint

outer-valid       -> test，只在最后评估一次
```

规则：

- 训练只用 `inner_train`，不碰 `outer-valid`；
- 每个 epoch 在 `inner_valid` 上计算 UAR/loss 等指标；
- 选 **inner-valid UAR 最高** 的 checkpoint，并列取最早 epoch；
- 用该 checkpoint 在 `outer-valid` 上评估一次；
- 不做 final retrain。

`earlystop` 是 checkpoint selection，不设置 patience 提前终止训练；默认扫描到各数据集的 `max_epochs`。

| 数据集 | 类别数 | fold | 默认 target_length | 默认 max_epochs |
|---|---:|---|---:|---:|
| EMO-DB | 7 | 1–10 | 640 | 80 |
| IEMOCAP-5531 | 4 | 1–5 | 1024 | 20 |
| RAVDESS-Speech | 8 | 0–4 | 512 | 80 |
| RAVDESS-Song | 6 | 0–4 | 640 | 80 |

- EMO-DB：10-fold speaker-independent；主结果用 535 条 OOF 预测池化。
- 其他数据集：fold 内评估，再对 folds/seeds 求 mean±std。
- 默认 `--norm-mode audioset`；`--norm-mode fold` 使用每折 `inner_train` 的有效帧统计量。

## 2. 数据与环境

```bash
export REPO=/root/songhao/ProJ/EAT2EMO
export SER_DATA_ROOT=/root/songhao/datasets/SER_DATASET
export SER_OUTPUT_ROOT=/root/songhao/experiments/EAT2EMO
export SER_PRETRAINED=/root/songhao/ProJ/resources/EAT_Weight/EAT-base_epoch30_pt.pt
export FAIRSEQ_ROOT=/root/songhao/ProJ/fairseq
export PYTHON_BIN=/root/miniconda3/envs/eat2emo/bin/python
export PYTHONDONTWRITEBYTECODE=1
```

生成/审计 manifests（首次）：

```bash
cd "$REPO"
$PYTHON_BIN tools/prepare_ser_manifests.py \
  --datasets-root "$SER_DATA_ROOT" \
  --manifest-root "$SER_DATA_ROOT" \
  --output "$SER_DATA_ROOT/manifests"
```

每个 fold 需要 `inner_train`、`inner_valid`、`train`、`valid` 四套 TSV/LBL。`valid` 是 outer-valid/test。

### 2.1 fold 归一化统计量

`norm_stats/<dataset>/selection/fold_XX.json` 保存 `inner_train` 的统计量，供 earlystop 训练和评估使用。目录名 `selection` 是历史命名，仅表示 inner-train 统计量。

```bash
cd "$REPO"
$PYTHON_BIN tools/compute_ser_norm_stats.py --dataset emodb --all-folds
```

生成后检查 JSON 的 `split == "inner_train"`、`stage == "selection"`。

### 2.2 特征缓存

可选：预生成 `inner_train` 的 fbank 缓存，避免首个 epoch 冷启动。

```bash
$PYTHON_BIN tools/precompute_ser_features.py --dataset emodb --all-folds
```

缓存位于 `$SER_DATA_ROOT/feature_cache/<dataset>/tl<T>_stats<hash>/`；hash 由 norm stats 文件内容决定。

## 3. 单 fold 训练

先 dry-run：

```bash
cd "$REPO"
bash scripts/finetuning_SER.sh \
  --dataset emodb --fold 1 --seed 3949 --dry-run
```

确认：

- `dataset.train_subset=inner_train`
- `dataset.valid_subset=inner_valid`
- `dataset.disable_validation=false`
- `max_epochs` / `max_updates` / `warmup_updates` 正确
- `checkpoint_best.pt` 是 eval checkpoint

正式运行：

```bash
bash scripts/finetuning_SER.sh \
  --dataset emodb --fold 1 --seed 3949 \
  --norm-mode fold --num-workers 12 \
  --wandb-project eat2emo-ser
```

可用 `--max-epochs N` 覆盖数据集默认扫描上限（smoke 用）。

## 4. W&B

```text
run name: <dataset>-earlystop-foldXX-seed<SEED>-<checkpoint stem>
group:    <dataset>-earlystop-<config>
job_type: earlystop
```

W&B 记录 `inner_valid` 的 loss/accuracy/WAR/UAR/macro-F1 和训练曲线；最终 outer-valid 指标在本地 `results/metrics.json`。

## 5. 输出目录

```text
${SER_OUTPUT_ROOT}/{dataset}/earlystop/{config}/seed_N/fold_XX/
├── run_manifest.json
├── train.log
├── checkpoint_best.pt
├── earlystop_summary.json
├── eval.log
└── results/
    ├── metrics.json
    ├── predictions.npz
    ├── paths.txt
    └── confusion_matrix.csv
```

`earlystop_summary.json` 包含：

- `max_epochs`, `best_epoch`, `best_uar`
- 逐 epoch `loss/accuracy/war/uar/macro_f1`
- `tie_break=earliest_epoch`

注意数值尺度：

- `earlystop_summary.json` / `train.log` 的 UAR 是 **0–100**
- `metrics.json` / summary 的 UAR 是 **0–1**

## 6. 批量运行

```bash
# 全部数据集 × 默认 seeds × 全 folds，并自动汇总
bash scripts/run_all_SER.sh \
  --norm-mode fold --num-workers 12 --wandb-project eat2emo-ser
```

支持：

```text
--datasets LIST
--seeds LIST|all
--folds LIST|all
--norm-mode audioset|fold
--max-epochs N
--num-workers N
--wandb-project NAME
--data-root PATH
--output-root PATH
--config-name NAME
--dry-run
```

部分 fold 子集会跳过自动 summary。

## 7. 手动评估

自动训练结束后会评估 best checkpoint。需要补评时直接调用：

```bash
$PYTHON_BIN evaluation/eval_ser.py \
  --data "$SER_DATA_ROOT/manifests/emodb/fold_01" \
  --checkpoint "$SER_OUTPUT_ROOT/emodb/earlystop/<config>/seed_3949/fold_01/checkpoint_best.pt" \
  --model-dir "$REPO" \
  --output "$SER_OUTPUT_ROOT/emodb/earlystop/<config>/seed_3949/fold_01/results" \
  --dataset emodb --fold 1 --seed 3949 \
  --selected-epoch <best_epoch> --max-epochs 80 \
  --norm-mode fold \
  --norm-stats-path "$SER_DATA_ROOT/norm_stats/emodb/selection/fold_01.json"
```

`<best_epoch>` 从同目录 `earlystop_summary.json` 读取。已有 artifacts 时加 `--overwrite`。

## 8. 汇总

所有 fold 完成后：

```bash
$PYTHON_BIN evaluation/summarize_ser.py \
  "$SER_OUTPUT_ROOT/emodb/earlystop/<config>" \
  --dataset emodb \
  --config <config> \
  --output "$SER_OUTPUT_ROOT/emodb/earlystop/<config>/summary.json"
```

输出：

```text
summary.json
summary_folds.csv
```

汇总会校验：

- protocol 必须为 `earlystop`
- `selected_epoch <= max_epochs`
- `trained_epochs == selected_epoch`
- 同一 config 下 `max_epochs` 一致
- outer-valid 覆盖完整、无重复

## 9. 运行原则

- batch size 固定 32；`ceil(inner_train_samples/32) × max_epochs` 个 updates，warmup 为 10%。
- 音频解码失败、空音频、无效样本立即终止，不跳过、不补位。
- `outer-valid` 不参与训练和选模；只在 best checkpoint 上评估一次。
- 单 fold 缺类时，UAR/Macro-F1 只对该 fold 实际出现的类计算；EMO-DB OOF 主指标使用完整 7 类。
- 旧协议结果目录保留为历史数据，不参与 earlystop summary。
