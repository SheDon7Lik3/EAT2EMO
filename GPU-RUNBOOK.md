# EAT2EMO GPU 运行手册

本文只记录 GPU 服务器上的连接、同步、运行、smoke 和排障。协议与指标口径见 [SER_FINETUNE_GUIDE.md](SER_FINETUNE_GUIDE.md)，环境版本见 [ENVIRONMENT.md](ENVIRONMENT.md)。

## 1. 服务器与资源

| 项目 | 值 |
|---|---|
| 连接 | `ssh -p 2222 root@127.0.0.1`（本机隧道） |
| 代码目录 | `/root/songhao/ProJ/EAT2EMO` |
| 数据根 | `/root/songhao/datasets/SER_DATASET` |
| 实验输出 | `/root/songhao/experiments/EAT2EMO` |
| conda 环境 | `/root/miniconda3/envs/eat2emo` |
| Fairseq | `/root/songhao/ProJ/fairseq` |
| GPU | 1 × RTX 4090，单卡串行 |
| 预训练权重 | `/root/songhao/ProJ/resources/EAT_Weight/EAT-base_epoch30_pt.pt` |
| AS-2M 权重 | `/root/songhao/ProJ/resources/EAT_Weight/EAT-base_epoch30_ft_encoder.pt` |

进入服务器先只读确认：

```bash
hostname
whoami
date
nvidia-smi
uptime
df -h /root
```

## 2. 操作边界

- 先只读，再写入；`rm`/`mv` 前先确认绝对路径。
- 不要全局 `pkill python`；只按 PID 或 screen 会话结束自己的任务。
- 单 GPU 串行，不要并行跑多个训练。
- 不覆盖已有实验目录；输出根按项目/协议隔离。
- 不在服务器上手改代码；本地改完 rsync。
- 数据、权重、实验输出、日志不进 Git。
- 旧协议实验目录和 `norm_stats` 保留为历史数据，不参与 earlystop 结果。

## 3. 代码同步

服务器是 rsync 副本，不是 git 仓库。本地改完、测试通过后同步：

```bash
rsync -ah --stats \
  --exclude='.git' \
  --exclude='datasets' \
  --exclude='checkpoints' \
  --exclude='results' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  --exclude='_backups' \
  --exclude='tb' \
  --exclude='wandb' \
  --exclude='multirun' \
  --exclude='*.png' \
  -e "ssh -p 2222 -o BatchMode=yes" \
  /Users/metro/Downloads/EAT2EMO/ \
  root@127.0.0.1:/root/songhao/ProJ/EAT2EMO/
```

同步后核对：

```bash
ssh -p 2222 root@127.0.0.1 '
  cd /root/songhao/ProJ/EAT2EMO
  sha256sum tools/ser_protocol.py evaluation/eval_ser.py evaluation/summarize_ser.py
  test ! -d datasets && test ! -d checkpoints && test ! -d .git \
    && echo "no data/weights/git"
'
```

## 4. 环境变量与预检

每个新 shell / screen 会话先执行：

```bash
export REPO=/root/songhao/ProJ/EAT2EMO
export SER_DATA_ROOT=/root/songhao/datasets/SER_DATASET
export SER_OUTPUT_ROOT=/root/songhao/experiments/EAT2EMO
export SER_PRETRAINED=/root/songhao/ProJ/resources/EAT_Weight/EAT-base_epoch30_pt.pt
export FAIRSEQ_ROOT=/root/songhao/ProJ/fairseq
export PYTHON_BIN=/root/miniconda3/envs/eat2emo/bin/python
export PYTHONDONTWRITEBYTECODE=1
```

不要用系统默认 `python`（3.13），必须用 `$PYTHON_BIN` 或 `conda activate eat2emo`。

预检：

```bash
cd "$REPO"
$PYTHON_BIN -V
$PYTHON_BIN -c "import torch; print(torch.__version__, torch.cuda.is_available())"
$PYTHON_BIN -c "import fairseq; print(fairseq.__version__)"
```

## 5. 数据与 manifest

四个数据集：

| 数据集 | WAV 数 | 类别 | target_length | fold | max_epochs |
|---|---:|---:|---:|---|---:|
| EMO-DB | 535 | 7 | 640 | 1–10 | 80 |
| IEMOCAP-5531 | 5531 | 4 | 1024 | 1–5 | 20 |
| RAVDESS-Speech | 1440 | 8 | 512 | 0–4 | 80 |
| RAVDESS-Song | 1012 | 6 | 640 | 0–4 | 80 |

首次生成/审计 manifests：

```bash
cd "$REPO"
$PYTHON_BIN tools/prepare_ser_manifests.py \
  --datasets-root "$SER_DATA_ROOT" \
  --manifest-root "$SER_DATA_ROOT" \
  --output "$SER_DATA_ROOT/manifests"
```

每个 fold 需要：

```text
inner_train.tsv/lbl   # 训练
inner_valid.tsv/lbl   # 选 best checkpoint
train.tsv/lbl         # 完整 outer-train（历史/统计用）
valid.tsv/lbl         # outer-valid/test
```

抽查：

```bash
ls "$SER_DATA_ROOT/manifests/emodb/fold_01"
head -3 "$SER_DATA_ROOT/manifests/emodb/fold_01/inner_train.tsv"
```

## 6. fold 归一化与特征缓存

`--norm-mode fold` 前，先生成对应 stage 的统计量：

```bash
for ds in emodb iemocap ravdess-speech ravdess-song; do
  # earlystop
  $PYTHON_BIN tools/compute_ser_norm_stats.py --dataset "$ds" --all-folds --stage selection
  # fixed
  $PYTHON_BIN tools/compute_ser_norm_stats.py --dataset "$ds" --all-folds --stage fixed
done
```

输出：

```text
$SER_DATA_ROOT/norm_stats/<dataset>/selection/fold_XX.json   # inner_train
$SER_DATA_ROOT/norm_stats/<dataset>/fixed/fold_XX.json       # train
```

不要重写/重命名已有文件，否则 feature cache 的 stats hash 会失效。

可选预生成训练特征缓存：

```bash
for ds in emodb iemocap ravdess-speech ravdess-song; do
  $PYTHON_BIN tools/precompute_ser_features.py --dataset "$ds" --all-folds --stage selection
  $PYTHON_BIN tools/precompute_ser_features.py --dataset "$ds" --all-folds --stage fixed
done
```

selection 缓存 `inner_train`，fixed 缓存完整 `train`。

## 7. 单 fold 训练

先 dry-run：

```bash
cd "$REPO"
# earlystop
bash scripts/finetuning_SER.sh \
  --dataset ravdess-speech --fold 0 --seed 3949 \
  --protocol earlystop --norm-mode fold --dry-run

# fixed
bash scripts/finetuning_SER.sh \
  --dataset ravdess-speech --fold 0 --seed 3949 \
  --protocol fixed --fixed-epochs 60 --norm-mode fold --dry-run
```

确认输出：

- earlystop：`inner_train` / `inner_valid`、`disable_validation=false`、`checkpoint_best.pt`
- fixed：`train`、`disable_validation=true`、`checkpoint_final.pt`

正式运行：

```bash
# earlystop
bash scripts/finetuning_SER.sh \
  --dataset ravdess-speech --fold 0 --seed 3949 \
  --protocol earlystop --norm-mode fold --num-workers 12 \
  --wandb-project eat2emo-ser

# fixed
bash scripts/finetuning_SER.sh \
  --dataset ravdess-speech --fold 0 --seed 3949 \
  --protocol fixed --fixed-epochs 60 --norm-mode fold --num-workers 12 \
  --wandb-project eat2emo-ser
```

AS-2M 权重：先 `export SER_PRETRAINED=...ft_encoder.pt`，命令同上。

`--max-epochs N` 只用于 earlystop；`--fixed-epochs N` 只用于 fixed；smoke 用 1。

## 8. 输出目录

```text
# earlystop
${SER_OUTPUT_ROOT}/{dataset}/earlystop/{config}/seed_N/fold_XX/
├── checkpoint_best.pt
├── earlystop_summary.json
└── results/...

# fixed
${SER_OUTPUT_ROOT}/{dataset}/fixed/{config}/seed_N/fold_XX/
├── checkpoint_final.pt
└── results/...
```

说明：

- earlystop：`checkpoint_best.pt` 是 inner-valid UAR 最优 checkpoint，`earlystop_summary.json` 含逐 epoch 指标；
- fixed：`checkpoint_final.pt` 是最后一个 checkpoint；
- 两种协议的 `results/metrics.json` 都是 outer-valid 单次评估。

## 9. W&B

```text
run name: <dataset>-<protocol>-foldXX-seed<SEED>-<checkpoint stem>
group:    <dataset>-<protocol>-<config>
job_type: earlystop / fixed
```

示例：

```bash
# earlystop
SER_OUTPUT_ROOT=/root/songhao/experiments/EAT2EMO \
  bash scripts/finetuning_SER.sh \
    --dataset ravdess-speech --fold 0 --seed 3949 \
    --protocol earlystop --norm-mode fold --num-workers 12 \
    --wandb-project eat2emo-ser

# fixed
SER_OUTPUT_ROOT=/root/songhao/experiments/EAT2EMO \
  bash scripts/finetuning_SER.sh \
    --dataset ravdess-speech --fold 0 --seed 3949 \
    --protocol fixed --fixed-epochs 60 --norm-mode fold --num-workers 12 \
    --wandb-project eat2emo-ser
```

W&B 查看训练曲线；earlystop 另有 `inner_valid` 的 loss/accuracy/WAR/UAR/macro-F1。

## 10. 监控

```bash
tail -n 50 -f "$SER_OUTPUT_ROOT/<dataset>/<protocol>/<config>/seed_3949/fold_XX/train.log"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
```

earlystop 结束后检查：

```bash
D="$SER_OUTPUT_ROOT/<dataset>/earlystop/<config>/seed_3949/fold_XX"
cat "$D/earlystop_summary.json" | head -40
cat "$D/results/metrics.json" | head -40
```

fixed 结束后检查：

```bash
D="$SER_OUTPUT_ROOT/<dataset>/fixed/<config>/seed_3949/fold_XX"
cat "$D/results/metrics.json" | head -40
```

## 11. 批量运行

默认全部数据集、5 seeds、全 folds、earlystop：

```bash
cd "$REPO"
bash scripts/run_all_SER.sh \
  --protocol earlystop --norm-mode fold --num-workers 12 \
  --wandb-project eat2emo-ser

# fixed
bash scripts/run_all_SER.sh \
  --protocol fixed --fixed-epochs 60 --norm-mode fold --num-workers 12 \
  --wandb-project eat2emo-ser
```

常用参数：

```text
--datasets LIST
--seeds LIST|all
--folds LIST|all
--protocol earlystop|fixed
--norm-mode audioset|fold
--max-epochs N       # earlystop
--fixed-epochs N     # fixed
--batch-size N       # default 32
--num-workers N
--wandb-project NAME
--data-root PATH
--output-root PATH
--config-name NAME
--dry-run
```

部分 fold 子集会跳过自动 summary。

后台运行：

```bash
nohup bash scripts/run_all_SER.sh \
  --protocol earlystop --norm-mode fold --num-workers 12 \
  --wandb-project eat2emo-ser \
  > run_all_earlystop.log 2>&1 &
```

## 12. Smoke

使用独立输出根：

```bash
export SER_OUTPUT_ROOT=/root/songhao/experiments/EAT2EMO_smoke

# earlystop
bash scripts/finetuning_SER.sh \
  --dataset emodb --fold 1 --seed 3949 \
  --protocol earlystop --norm-mode fold --max-epochs 1

# fixed
bash scripts/finetuning_SER.sh \
  --dataset emodb --fold 1 --seed 3949 \
  --protocol fixed --fixed-epochs 1 --norm-mode fold
```

检查：

```text
earlystop: checkpoint_best.pt / earlystop_summary.json / results/metrics.json
fixed:     checkpoint_final.pt / results/metrics.json
```

`_maxep1` / `_fixedep1` 会进入 config 名，与正式结果隔离。

## 13. 手动补评

自动流程已包含 eval。补评直接调用：

```bash
# earlystop
D="$SER_OUTPUT_ROOT/emodb/earlystop/<config>/seed_3949/fold_01"
BEST=$($PYTHON_BIN -c "import json; print(json.load(open('$D/earlystop_summary.json'))['best_epoch'])")
MAX=$($PYTHON_BIN -c "import json; print(json.load(open('$D/earlystop_summary.json'))['max_epochs'])")

$PYTHON_BIN evaluation/eval_ser.py \
  --data "$SER_DATA_ROOT/manifests/emodb/fold_01" \
  --checkpoint "$D/checkpoint_best.pt" \
  --model-dir "$REPO" \
  --output "$D/results" \
  --dataset emodb --protocol earlystop --fold 1 --seed 3949 \
  --selected-epoch "$BEST" --max-epochs "$MAX" \
  --norm-mode fold \
  --norm-stats-path "$SER_DATA_ROOT/norm_stats/emodb/selection/fold_01.json" \
  --overwrite

# fixed
D="$SER_OUTPUT_ROOT/emodb/fixed/<config>/seed_3949/fold_01"
$PYTHON_BIN evaluation/eval_ser.py \
  --data "$SER_DATA_ROOT/manifests/emodb/fold_01" \
  --checkpoint "$D/checkpoint_final.pt" \
  --model-dir "$REPO" \
  --output "$D/results" \
  --dataset emodb --protocol fixed --fold 1 --seed 3949 \
  --fixed-epochs 60 \
  --norm-mode fold \
  --norm-stats-path "$SER_DATA_ROOT/norm_stats/emodb/fixed/fold_01.json" \
  --overwrite
```

## 14. 汇总

```bash
# earlystop
$PYTHON_BIN evaluation/summarize_ser.py \
  "$SER_OUTPUT_ROOT/emodb/earlystop/<config>" \
  --dataset emodb --protocol earlystop --config <config> \
  --output "$SER_OUTPUT_ROOT/emodb/earlystop/<config>/summary.json"

# fixed
$PYTHON_BIN evaluation/summarize_ser.py \
  "$SER_OUTPUT_ROOT/emodb/fixed/<config>" \
  --dataset emodb --protocol fixed --config <config> --fixed-epochs 60 \
  --output "$SER_OUTPUT_ROOT/emodb/fixed/<config>/summary.json"
```

输出 `summary.json` 和 `summary_folds.csv`。

## 15. 常见问题

| 现象 | 处理 |
|---|---|
| `Missing inner_train file` | manifests 未生成或不完整，重跑 §5 |
| `Missing norm stats` | 先生成 §6 对应 stage 的统计量 |
| `Python 3.13` 报错 | 没导出 `PYTHON_BIN`，重新执行 §4 |
| `--max-epochs` / `--fixed-epochs` 非法 | 必须是正整数 |
| `--max-epochs` 用在 fixed | 只能用于 earlystop |
| `Refusing to overwrite` | 该 seed/fold 结果已存在；换输出根或清理旧目录 |
| `Training did not produce checkpoint_best.pt` | earlystop 训练失败或 inner-valid UAR 未记录；看 `train.log` |
| `Training did not produce checkpoint_last.pt` | fixed 训练失败；看 `train.log` |
| `Expected inner-valid metrics` | earlystop `train.log` 缺 epoch 或格式不是 JSON |
| `selected_epoch exceeds max_epochs` | earlystop summary 与 eval 参数不一致 |
| `results` 已存在 | 手动补评加 `--overwrite` |
| GPU 闲置但任务在跑 | 可能在 validation/checkpoint 写入；看 `train.log` 和 `nvidia-smi` |

## 16. 正式运行检查清单

- [ ] 代码已 rsync，关键文件 SHA256 一致
- [ ] `PYTHON_BIN` 指向 eat2emo
- [ ] manifests 含 `train`/`valid` 和 `inner_train`/`inner_valid`
- [ ] fold 模式下对应 stage 的 norm stats 存在
- [ ] dry-run 输出 `protocol=earlystop` 或 `protocol=fixed`，split/checkpoint 正确
- [ ] smoke 产出对应 checkpoint 和 `results/metrics.json`
- [ ] 正式结果在独立 `SER_OUTPUT_ROOT`
- [ ] 全 fold 完成后 summary 校验通过
- [ ] W&B job_type 为 `earlystop` 或 `fixed`
