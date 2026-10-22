# EAT2EMO GPU 运行手册

本文记录 EAT2EMO 在 GPU 服务器上的实际入口、固定资源、已完成的部署/环境状态和运行步骤。它是操作说明，不是“已训练完成”的证明。2026-10-19 的进展：代码已由本地 rsync 部署到服务器，`eat2emo` 环境已创建并通过预检；**manifest 尚未生成，训练尚未启动**。注意：服务器代码是**早期同步版本**；之后本地修复了 IEMOCAP extractor、清理了失效的上游脚本/配置/评测代码并更新了文档，服务器副本尚未重新同步。

## 文档分工

| 文档 | 负责内容 |
|---|---|
| [README.md](README.md) | 项目入口、上游 EAT、SER 快速入口 |
| [ENVIRONMENT.md](ENVIRONMENT.md) | 环境安装、版本固定、当前 `eat2emo` 实况（版本唯一来源） |
| [SER_FINETUNE_GUIDE.md](SER_FINETUNE_GUIDE.md) | SER 协议、命令语义、指标口径 |
| 本文 | GPU 服务器连接、rsync 部署、运行、smoke 与排障 |

> 本文只保留“服务器操作”相关内容；环境版本以 ENVIRONMENT.md §B 为准，协议与指标以 SER_FINETUNE_GUIDE.md 为准。

---

## 1. 当前状态（2026-10-19）

### 1.1 一句话结论

- ✅ **数据与预训练权重已就绪**：`SER_DATASET` 四个子集数量和布局与协议一致，权重与本地文件 SHA256 相同。
- ⚠️ **代码副本已部署但已过期**：服务器 `/root/songhao/ProJ/EAT2EMO` 是 2026-10-19 早期 rsync 副本（60 files / 约 4.0 MB，无 `.git`）；本地之后修复了 IEMOCAP extractor、清理了失效的上游脚本/配置/评测代码并更新了文档。**训练前需要重新 rsync。**
- ✅ **`eat2emo` 环境已验证**：Python 3.10.20、torch 2.1.2+cu121、numpy 1.26.3、sklearn 1.3.2、fairseq 0.12.2、timm 0.9.12、transformers 4.51.3；`pip check` 无 broken requirements。旧 `emoat` 保留作回退。
- ⏳ **Manifest 与实验输出尚未生成**：`SER_DATASET/manifests` 和 `/root/songhao/experiments/EAT2EMO` 都还不存在。
- ✅ **Git 同步已完成**：本地 `main` 与 GitHub `origin/main` 一致、工作区 clean；服务器仍是早期 rsync 副本，不能 `git pull`，训练前需重新 rsync 或按 §3.2 切回 Git。

当前环境与代码已经就绪，但训练前仍必须先完成：生成 manifest（§5）→ dry-run/smoke（§6）→ 再正式批量（§7）。

### 1.2 固定资源与实测状态

| 项目 | 实测值（2026-10-19） | 状态 / 说明 |
|---|---|---|
| 连接入口 | `ssh -p 2222 root@127.0.0.1`（本机 Cactus 隧道） | ✅ 当前可用 |
| 直连 fallback | 本机 `~/.ssh/config` 的 `eaglelab` | ⚠️ 检查时端口 refused，需 VPN/网络/端口确认；地址不写入仓库 |
| 执行环境 | Jupyter 作业容器，hostname `jupyter-...` | ⚠️ 容器重启后 hostname/PID 会变，不要写进脚本 |
| 操作系统 | Ubuntu 22.04.4 LTS | ✅ |
| GPU | 1 × NVIDIA GeForce RTX 4090，24564 MiB，driver 595.71.05 | ✅ 检查时 0% 利用率、无计算进程 |
| CPU / 内存 | 88 vCPU / 550 GiB；load average 76.86（检查时） | ⚠️ 共享主机，跑前用 `uptime` 复查 |
| 磁盘 | `/root` 7.3T，已用 383G，可用 6.9T；inode 使用 4% | ✅ |
| 代码目录 | `/root/songhao/ProJ/EAT2EMO`，早期 rsync 副本（服务器 60 files / 4.0MB；本地已更新，未同步） | ⚠️ 需重新 rsync |
| conda 环境 | `/root/miniconda3/envs/eat2emo`，Python 3.10.20；旧 `emoat` 保留作回退 | ✅ 已验证（§4.2） |
| Fairseq | `/root/songhao/ProJ/fairseq`，0.12.2，HEAD `3d262bb2`，`import fairseq` 已验证 | ✅ |
| 预训练权重 | `/root/songhao/ProJ/resources/EAT_Weight/EAT-base_epoch30_pt.pt`，1.4G | ✅ SHA256 见 §1.3 |
| 数据根 | `/root/songhao/datasets/SER_DATASET` | ✅ 535 / 5531 / 1440 / 1012 |
| Manifest | `/root/songhao/datasets/SER_DATASET/manifests` | ⏳ 不存在，待生成 |
| 实验输出 | `/root/songhao/experiments/EAT2EMO` | ⏳ 不存在，首次运行会创建 |
| GitHub | `https://github.com/SheDon7Lik3/EAT2EMO.git`，HTTP 200，服务器可 `ls-remote` | ✅ 本地 `main` 与 `origin/main` 一致；服务器为过期 rsync 副本 |
| 后台工具 | `screen` 4.09、`rsync`、`git`、`curl`、`sha256sum` 可用；`tmux` 缺失 | ✅ 用 `screen` 或 `nohup` |
| 系统默认 python | `/root/miniconda3/bin/python` 是 3.13.13 | ⚠️ 不要用它跑本项目，必须显式用 eat2emo |

> 完整包版本与验证结果见 [ENVIRONMENT.md](ENVIRONMENT.md) §B；本表只列运行相关状态。

### 1.3 数据与权重明细

| 数据集 | 服务器目录 | WAV 数 | 类别 | 默认 target_length | fold |
|---|---|---:|---:|---:|---|
| EMO-DB | `SER_DATASET/EMODB` | 535 | 7 | 640 | 1–10 |
| IEMOCAP-5531 | `SER_DATASET/IEMOCAP-5531/Session1..5` | 5531 | 4 | 1024 | 1–5 |
| RAVDESS-Speech | `SER_DATASET/RAVDESS-Speech/Actor_01..24` | 1440 | 8 | 512 | 0–4 |
| RAVDESS-Song | `SER_DATASET/RAVDESS-Song/Actor_01..24`（无 Actor_18） | 1012 | 6 | 640 | 0–4 |

四个数据目录合计约 **1.2G**，服务器上已存在，不需要上传。

IEMOCAP 核对结果：

- `metadata.csv` 表头：`utterance_id,emotion,session,speaker,wav`，数据行 5531。
- 标签计数：`ang=1103`、`hap=1636`、`neu=1708`、`sad=1084`。
- Session 计数：`1=1085`、`2=1023`、`3=1151`、`4=1031`、`5=1241`。
- `wav` 列里是导出机器的旧路径（`/Users/metro/...`），**生成 manifest 时会被忽略**；脚本按 `IEMOCAP-5531/SessionN/<utterance_id>.wav` 定位，不要手工改这一列。
- 抽检 EMODB / IEMOCAP / RAVDESS-Speech / RAVDESS-Song 各一个 WAV，均为 16000 Hz、单声道；全量格式由 manifest 生成器再次审计。

预训练权重：

```text
path:   /root/songhao/ProJ/resources/EAT_Weight/EAT-base_epoch30_pt.pt
size:   1414209018 bytes (约 1.4G)
sha256: 91228f599658bbdac5afa02a8a60e0eea1fc8960cab12feab6f2d555adf03816
```

本地 `checkpoints/EAT-base_epoch30_pt.pt` 的 SHA256 与服务器完全一致，**不需要重复上传**。如果重新下载，先核对这个哈希。

> 旧文档里的 `/root/songhao/datasets/ser_datasets`、`/root/songhao/datasets/checkpoints` 当前都不存在；以本手册的 `SER_DATASET` 和 `resources/EAT_Weight` 为准。

---

## 2. 连接与操作边界

### 2.1 连接方式

当前实测可用的是本机已有隧道：

```bash
ssh -p 2222 root@127.0.0.1
```

本机 `127.0.0.1:2222` 由 Cactus 隧道提供（检查时是 `node` 进程监听）。它不属于本项目管理对象；不要随意结束该进程。

本机 SSH 配置里还有 `Host eaglelab` 的直连配置，但检查时其端口连接被拒绝。需要走直连时，先确认 VPN/网络/端口，再用 `ssh eaglelab`；**具体 HostName/User/Port 只保留在本机 `~/.ssh/config`，不写进仓库**。

进入服务器后先做只读确认：

```bash
hostname
whoami
date
nvidia-smi
uptime
df -h /root
```

### 2.2 操作边界

- **先只读，再写入**：任何 `rm`、`mv`、`pip install`、`git pull` 前先确认目标路径和当前轮次。
- **不要全局 `pkill python` / `pkill -9`**：只按 `screen` 会话、PID 或 `nvidia-smi` 里的进程结束自己的任务。
- **单 GPU 串行**：服务器只暴露 1 张 RTX 4090，且没有 Slurm/队列调度；所有 fold 需要手动串行，不要并行跑多个训练。
- **不要覆盖别人的目录**：`SER_OUTPUT_ROOT` 用本项目自己的路径；已有正式结果不要覆盖。
- **不要在服务器上改代码**：本地改 → 测试 → 重新 rsync（当前方式）或 commit/push（长期方式）。服务器当前是 rsync 副本，不是 git 仓库。
- **数据/权重不进 Git**：`SER_DATASET`、`EAT_Weight`、实验输出、日志都不提交。
- **不要使用本地旧 manifest**：本地 `datasets/manifests/*` 是 `train/eval/test` 旧协议（IEMOCAP 还是 6877 条），与当前 `nested/fixed` 协议不兼容。

---

## 3. 代码同步（当前为 rsync 副本）

### 3.1 当前部署方式（2026-10-19）

服务器 `/root/songhao/ProJ/EAT2EMO` 是本地工作区的**早期**代码副本，由 rsync 同步：

- 服务器副本：60 个文件 / 约 4.0 MB；
- 本地之后已更新：修复了 IEMOCAP extractor、清理了失效的上游脚本/配置/评测代码、更新了文档；不要用文件数判断是否一致，训练前重新 rsync；
- 排除 `datasets/`、`checkpoints/`、`results/`、`.git/`、缓存和备份目录；
- 服务器这份不是 git 仓库，不能 `git pull`；
- 属主已归一为 `root:root`，`GPU-RUNBOOK.md` 权限 644；
- 同步时的 `requirements.txt`、`tools/ser_protocol.py`、`scripts/run_all_SER.sh` SHA256 与本地一致，但 `tools/extract_iemocap_parquet.py` 和文档之后已更新；
- 训练前必须重新 rsync，确认服务器与本地 HEAD 一致。

更新代码时从本地重新 rsync，不要把数据/权重带上去：

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
  -e "ssh -p 2222 -o BatchMode=yes" \
  /Users/metro/Downloads/EAT2EMO/ \
  root@127.0.0.1:/root/songhao/ProJ/EAT2EMO/
```

同步后核对：

```bash
ssh -p 2222 root@127.0.0.1 '
  cd /root/songhao/ProJ/EAT2EMO
  find . -type f | wc -l
  du -sh .
  test ! -d datasets && test ! -d checkpoints && test ! -d .git \
    && echo "no data/weights/git"
'
```

### 3.2 长期方案：GitHub

GitHub 仍是长期代码源：`https://github.com/SheDon7Lik3/EAT2EMO.git`。

当前状态（2026-10-19）：

```text
GitHub origin/main: 最新提交（见 git log；已包含 SER 协议、extractor 修复、文档和上游清理）
本地 HEAD:          与 origin/main 一致，工作区 clean
服务器:             早期 rsync 副本，无 .git，需重新同步
```

如果要切回 Git：

```bash
# 本地
cd /Users/metro/Downloads/EAT2EMO
git add -A
git commit -m "SER protocol, runbook and environment docs"
git push origin main

# 服务器：不要直接在 rsync 副本上 git init/pull
cd /root/songhao/ProJ
mv EAT2EMO EAT2EMO.rsync.bak
git clone https://github.com/SheDon7Lik3/EAT2EMO.git EAT2EMO
cd EAT2EMO
git rev-parse HEAD
```

> 切换前，服务器以 rsync 副本运行；不要对副本执行 `git pull`。

### 3.3 同步边界

- 代码与数据分离：`SER_DATASET`、`EAT_Weight`、实验输出、日志都不进代码同步。
- 服务器上不要手改代码；改本地后重新 rsync，或按 §3.2 切到 Git。
- 每次同步后用文件数和关键文件 SHA256 复核。

---

## 4. 环境预检与当前环境

### 4.1 统一环境变量

```bash
export REPO=/root/songhao/ProJ/EAT2EMO
export SER_DATA_ROOT=/root/songhao/datasets/SER_DATASET
export SER_OUTPUT_ROOT=/root/songhao/experiments/EAT2EMO
export SER_PRETRAINED=/root/songhao/ProJ/resources/EAT_Weight/EAT-base_epoch30_pt.pt
export FAIRSEQ_ROOT=/root/songhao/ProJ/fairseq
export PYTHON_BIN=/root/miniconda3/envs/eat2emo/bin/python
export PYTHONDONTWRITEBYTECODE=1
```

> 系统默认 `python` 是 3.13.13，不是本项目环境。所有命令要么显式写 `$PYTHON_BIN`，要么先 `conda activate eat2emo`；在 `screen`/`nohup` 里建议直接用绝对路径。
>
> 旧 `emoat` 环境保留作回退，但不再作为本项目默认环境；`PYTHON_BIN` 必须指向 `eat2emo`。
>
> ⚠️ 如果不导出 `PYTHON_BIN`，`scripts/finetuning_SER.sh`、`scripts/run_all_SER.sh`、`scripts/eval_SER.sh` 会退回 PATH 里的 `python`（当前是 3.13.13），训练子进程也会用 3.13，直接失败。每个新 shell / `screen` session 都先执行本节 export。

### 4.2 必做预检（2026-10-19 已通过）

```bash
PY=/root/miniconda3/envs/eat2emo/bin/python
export PYTHONDONTWRITEBYTECODE=1

$PY -V
$PY -m pip --version
$PY -c "import numpy; print('numpy', numpy.__version__)"
$PY -c "import sklearn; print('sklearn', sklearn.__version__)"
$PY -c "import torch; print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
$PY -c "import fairseq; print('fairseq', fairseq.__version__, fairseq.__file__)"
$PY -c "import timm; print('timm', timm.__version__)"
$PY -c "import transformers; print('transformers', transformers.__version__)"
$PY -m pip check
nvidia-smi
```

期望值/版本唯一来源见 [ENVIRONMENT.md](./ENVIRONMENT.md) §B。关键要求：Python 3.10.x、pip 24.0、numpy 1.26.3、scikit-learn 1.3.2、torch 2.1.2+cu121、fairseq 0.12.2、timm 0.9.12、transformers 4.51.3；`pip check` 无 broken requirements。2026-10-19 已通过。

### 4.3 历史问题（旧 emoat，已解决）

旧 `emoat` 曾因 numpy 2.2.6 + scikit-learn 1.3.2 导致 `import sklearn` / `import fairseq` 失败；该环境保留作回退，不再使用。症状与修复见 [ENVIRONMENT.md](./ENVIRONMENT.md) §8。

### 4.4 环境构建记录

新环境 `eat2emo` 于 2026-10-19 按 [ENVIRONMENT.md](./ENVIRONMENT.md) §A 构建，并通过 §4.2 预检；详细包版本见 ENVIRONMENT §B。要点：`pip==24.0`、`cython<3`、`numpy==1.26.3`、`setuptools==69.5.1`、editable Fairseq 0.12.2、`pip check` 通过。

### 4.5 资源预检

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
uptime
free -h
df -h /root
df -i /root
```

2026-10-19 检查时 GPU 空闲；主机 load 约 76.86（88 vCPU），属于共享主机，开跑前必须复查。若 load 明显更高，先与服务器其他使用者协调。

---

## 5. 数据与 Manifest

### 5.1 目录约定

```text
SER_DATA_ROOT=/root/songhao/datasets/SER_DATASET
├── EMODB/
├── IEMOCAP-5531/
│   ├── metadata.csv
│   └── Session1/ ... Session5/
├── RAVDESS-Speech/
│   └── Actor_01/ ... Actor_24/
├── RAVDESS-Song/
│   └── Actor_01/ ... Actor_24/   # 无 Actor_18
└── manifests/                    # 首次生成，当前不存在
```

每个 fold 生成：

```text
train.tsv / train.lbl
valid.tsv / valid.lbl
inner_train.tsv / inner_train.lbl
inner_valid.tsv / inner_valid.lbl
label_descriptors.csv
split.json
```

### 5.2 生成 manifest（首次，服务器上执行）

```bash
cd "$REPO"
$PYTHON_BIN tools/prepare_ser_manifests.py \
  --datasets-root "$SER_DATA_ROOT" \
  --manifest-root "$SER_DATA_ROOT" \
  --output "$SER_DATA_ROOT/manifests"
```

期望输出类似：

```text
emodb: 535 samples, 10 folds
iemocap: 5531 samples, 5 folds
ravdess-speech: 1440 samples, 5 folds
ravdess-song: 1012 samples, 5 folds
```

说明：

- `--manifest-root` 会写到每个 TSV 的第一行，必须是**服务器**数据根；不要写本机路径。
- 生成器会强制审计样本数、标签分布、说话人/session 分组、outer/inner 零交叉、outer-valid 恰好覆盖一次、所有 WAV 为单声道 16 kHz；任一失败会直接终止，**不要绕过审计**。
- 重新生成会覆盖同路径 manifest；如果已有正式结果引用它们，不要随意重跑。
- 本地 `datasets/manifests/*` 是旧协议，**不要上传或拷贝到服务器**。

> 训练默认启用 fbank 特征缓存：首次 run 在 `${SER_DATA_ROOT}/feature_cache/<dataset>/...` 生成 `.npy`，后续 epoch/fold/seed 复用；`roll_aug/noise` 训练自动绕过。缓存与实时 fbank 的数值差异约 1e-7，正式结果不要混用 cached / non-cached；删除该目录可重建。

### 5.3 生成后抽查

```bash
ls "$SER_DATA_ROOT/manifests/emodb/fold_01"
head -3 "$SER_DATA_ROOT/manifests/emodb/fold_01/train.tsv"
cat "$SER_DATA_ROOT/manifests/emodb/fold_01/label_descriptors.csv"
$PYTHON_BIN -c "import json; d=json.load(open('$SER_DATA_ROOT/manifests/emodb/fold_01/split.json')); print(d['dataset'], d['fold'], d['audits'])"
```

确认 TSV 第一行是 `$SER_DATA_ROOT`，条目是相对路径（如 `EMODB/03a01Fa.wav`），`split.json` 里 `audits` 全为 `true`。

---

## 6. 单 fold 训练

### 6.1 先 dry-run

```bash
cd "$REPO"
bash scripts/finetuning_SER.sh \
  --dataset emodb --fold 1 --seed 3949 --protocol nested --dry-run
```

`--dry-run` 只校验输入并打印 JSON / `TRAIN:` / `EVAL:` 命令，不写实验目录。确认：

- `target_length`、`num_classes` 正确；
- `max_updates` / `warmup_updates` 由样本数自动计算；
- `dataset.valid_subset` 在 selection 阶段是 `inner_valid`，在 final/fixed 阶段是 `train`；
- `checkpoint.restore_file` 指向 fresh start（不是上一阶段 checkpoint）。

### 6.2 Nested（正式协议）

```bash
cd "$REPO"
export PYTHON_BIN=/root/miniconda3/envs/eat2emo/bin/python
export SER_DATA_ROOT=/root/songhao/datasets/SER_DATASET
export SER_OUTPUT_ROOT=/root/songhao/experiments/EAT2EMO
export SER_PRETRAINED=/root/songhao/ProJ/resources/EAT_Weight/EAT-base_epoch30_pt.pt
export FAIRSEQ_ROOT=/root/songhao/ProJ/fairseq

bash scripts/finetuning_SER.sh \
  --dataset emodb --fold 1 --seed 3949 --protocol nested
```

协议语义（selection/final/eval 各自做什么、哪些 split 可见）见 [SER_FINETUNE_GUIDE.md](SER_FINETUNE_GUIDE.md) §1 和 §7。本节只保留服务器上的操作命令。

也可以分阶段：

```bash
bash scripts/finetuning_SER.sh --dataset emodb --fold 1 --seed 3949 --protocol nested --stage select
bash scripts/finetuning_SER.sh --dataset emodb --fold 1 --seed 3949 --protocol nested --stage final
```

`final` 会读取 `selection/selection_summary.json`，并校验 dataset/protocol/config/fold/seed/pretrained 是否一致；不一致会拒绝。

### 6.3 Fixed（对照协议）

```bash
bash scripts/finetuning_SER.sh \
  --dataset emodb --fold 1 --seed 3949 --protocol fixed --fixed-epochs 60
```

`fixed` 只读 `train/valid`，固定训练 60 epoch，全程不验证，结束后只评估一次 `valid`。`--stage` 只适用于 nested，不要传给 fixed。

诊断曲线（每 epoch 在 outer-valid 上算 loss / accuracy / WAR / UAR / macro-F1）：

```bash
bash scripts/finetuning_SER.sh \
  --dataset emodb --fold 1 --seed 3949 --protocol fixed --fixed-epochs 20 \
  --diagnostic-validate-outer --wandb-project eat2emo-ser
```

该模式只允许 fixed；config/run 名会自动加 `_diagouter` / `-diagouter`。结果目录额外写 `diagnostic_history.json`，最终 `metrics.json` 标记为 diagnostic，`summarize_ser.py` 会忽略。**不要用这些曲线挑 epoch。**

### 6.4 输出目录与文件

```text
${SER_OUTPUT_ROOT}/{dataset}/{protocol}/{config}/seed_N/fold_XX/{selection|final|fixed}/
├── run_manifest.json          # 实际命令、样本数、调度、归一化、pretrained、diagnostic 标记
├── train.log                  # Fairseq JSON 日志（诊断模式含逐 epoch valid 指标）
├── checkpoint_last.pt
├── checkpoint_best.pt         # 仅 selection
├── checkpoint_best_outer_diagnostic.pt  # 仅诊断 fixed，由 outer-valid 选出，禁止正式使用
├── checkpoint_final.pt        # final/fixed，checkpoint_last 的硬链接
├── selection_summary.json     # 仅 selection
├── diagnostic_history.json    # 仅 --diagnostic-validate-outer
├── eval.log
└── results/
    ├── metrics.json
    ├── predictions.npz
    ├── paths.txt
    └── confusion_matrix.csv
```

默认 config 名是 `eat_base_sr16k_audiosetnorm_bs32`。非默认 epoch 会自动加后缀隔离：

- fixed 非 60 epoch → `..._fixedepN`
- fixed 诊断模式 → `..._diagouter`
- nested 非默认 selection 上限 → `..._selectmaxepN`

不要手工把 smoke 结果和正式 config 混在一起。

### 6.5 监控

```bash
tail -n 50 -f "$SER_OUTPUT_ROOT/emodb/nested/eat_base_sr16k_audiosetnorm_bs32/seed_3949/fold_01/final/train.log"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
```

诊断模式下，`train.log` 的 `valid_loss` / `valid_accuracy` / `valid_war` / `valid_uar` / `valid_macro_f1` 会逐 epoch 出现；完整汇总见同目录 `diagnostic_history.json`。W&B 中对应 run 的 job type 为 `fixed-diagnostic`，并带 `diagnostic-outer-valid` tag。

协议固定 `BATCH_SIZE=32`、fp16、Adam `5e-5`、wd `0.05`、cosine + 10% warmup、`min_lr=1e-6`；不要通过 CLI 改 batch size，输出目录和 `run_manifest.json` 都按这个记录。

---

## 7. 批量运行

### 7.1 先做 smoke（独立输出根）

在 §4.1 已导出环境变量的同一个 shell / `screen` session 中执行：

```bash
export SER_OUTPUT_ROOT=/root/songhao/experiments/EAT2EMO_smoke

# nested：1 个 fold、1 个 epoch 选模，再 1 个 epoch final + 1 次 valid 评估
bash scripts/finetuning_SER.sh \
  --dataset emodb --fold 1 --seed 3949 --protocol nested --selection-max-epochs 1

# fixed：1 个 fold、1 个 epoch
bash scripts/finetuning_SER.sh \
  --dataset emodb --fold 1 --seed 3949 --protocol fixed --fixed-epochs 1
```

smoke 会自动使用 `_selectmaxep1` / `_fixedep1` config 后缀，和正式结果隔离。检查 smoke 产出的 `metrics.json`、`predictions.npz`、`paths.txt`、`confusion_matrix.csv`、`eval.log` 都存在且可读。

### 7.1.1 生成 fold 归一化统计量（使用 `--norm-mode fold` 前）

```bash
cd "$REPO"
for ds in emodb iemocap ravdess-speech ravdess-song; do
  for stage in fixed selection final; do
    $PYTHON_BIN tools/compute_ser_norm_stats.py \
      --dataset "$ds" --all-folds --stage "$stage" --overwrite
  done
done
```

统计量只累计对应训练 split 的有效帧，输出到
`$SER_DATA_ROOT/norm_stats/<dataset>/<stage>/fold_XX.json`。

预生成训练特征缓存（可选，避免首个 epoch 冷启动）：

```bash
for ds in emodb iemocap ravdess-speech ravdess-song; do
  for stage in fixed selection final; do
    $PYTHON_BIN tools/precompute_ser_features.py \
      --dataset "$ds" --all-folds --stage "$stage"
  done
done
```

### 7.2 正式批量

长跑先开 `screen`（服务器没有 `tmux`）：

```bash
screen -S eat2emo
# detach：Ctrl-A 然后 D
# 重新连接：screen -r eat2emo
```

如果 `screen` 不可用，用 `nohup` 并记录 PID：

```bash
nohup bash scripts/run_all_SER.sh --protocol nested > run_all_nested.log 2>&1 &
echo $!
```

然后在 session 内执行：

```bash
export SER_OUTPUT_ROOT=/root/songhao/experiments/EAT2EMO
cd "$REPO"

# nested：4 个数据集，seeds 3949,42,2026,1024,666
bash scripts/run_all_SER.sh --protocol nested

# fixed：4 个数据集，seeds 3949,42,2026,1024,666，60 epoch
bash scripts/run_all_SER.sh --protocol fixed --fixed-epochs 60

# fixed + fold 归一化（需先生成 norm_stats）；12 workers 实测略快
bash scripts/run_all_SER.sh --protocol fixed --fixed-epochs 60 --norm-mode fold --num-workers 12
```

只跑部分数据集/seed/fold：

```bash
# 指定数据集、seeds、folds
bash scripts/run_all_SER.sh \
  --datasets emodb,ravdess-speech --seeds 3949,42 \
  --folds 1,2 --protocol nested

# 只跑一个 seed 的一个 fold
bash scripts/run_all_SER.sh \
  --datasets emodb --seeds 3949 --folds 1 --protocol nested

# 显式表示全部
bash scripts/run_all_SER.sh \
  --seeds all --folds all --protocol nested
```

`--seeds all` 是 5 个参考 seed；`--folds all` 是每个数据集自己的 full range。显式 fold 必须对选中的每个 dataset 合法，否则启动前报错。

先看命令不执行：

```bash
bash scripts/run_all_SER.sh --protocol nested --datasets emodb --seeds 3949 --folds 1 --dry-run
```

批量脚本会按 dataset 顺序串行执行 fold/seed。只有 `--folds all` 时才自动调用 `summarize_ser.py` 做严格汇总；显式 fold 子集会跳过自动汇总，per-fold 的 `metrics.json` 仍保留。单 GPU 下不要并行跑多个 `run_all_SER.sh`。

### 7.3 规模与时间

每个协议完整跑 5 seeds（3949,42,2026,1024,666）是 **125 个 fold-run**：

| 数据集 | folds | seeds | fold-runs |
|---|---:|---:|---:|
| EMO-DB | 10 | 5 | 50 |
| IEMOCAP | 5 | 5 | 25 |
| RAVDESS-Speech | 5 | 5 | 25 |
| RAVDESS-Song | 5 | 5 | 25 |
| **合计** | | | **125** |

nested 每个 fold-run 包含 selection + final 两次训练（共 250 次训练），fixed 是 125 次训练；两种协议都跑约 375 次训练。**本轮没有实测单次耗时**，不要根据历史文档估算；先跑 smoke 和 1 个正式 fold，记录 `train.log` 的 epoch 时间后再推算。

checkpoint 会保留 `last/best/final`，单个约 1.4G（参考预训练文件）；125 个 nested fold-run 的 checkpoint 可能达到数百 GB。`/root` 当前余量 7.0T，够用，但批量跑前后都要 `df -h /root`。

---

## 8. 评测与汇总

### 8.1 单 fold 正式评测

```bash
cd "$REPO"
bash scripts/eval_SER.sh --dataset emodb --fold 1 --seed 3949 --protocol nested
```

`eval_SER.sh` 会自动定位：

```text
${SER_OUTPUT_ROOT}/emodb/nested/eat_base_sr16k_audiosetnorm_bs32/seed_3949/fold_01/final/checkpoint_final.pt
```

并输出到同目录 `results/`。如果用了自定义 config 或 smoke config，显式传：

```bash
bash scripts/eval_SER.sh --dataset emodb --fold 1 --seed 3949 --protocol nested \
  --config eat_base_sr16k_audiosetnorm_bs32_selectmaxep1
```

注意：

- 正式评测只接受 `stage=final`（nested）或 `stage=fixed`（fixed），`split=valid`；
- 默认 `--norm-mode audioset` 使用 `mean=-4.268`、`std=4.569`；
- `--norm-mode fold` 使用 `$SER_DATA_ROOT/norm_stats/<dataset>/<stage>/fold_XX.json`，只归一化有效帧、padding 保持 0；
- 默认拒绝覆盖已有 `results/` 产物；只有确认要替换时才加 `--overwrite`。

### 8.2 汇总

```bash
cd "$REPO"
$PYTHON_BIN evaluation/summarize_ser.py \
  /root/songhao/experiments/EAT2EMO/emodb/nested/eat_base_sr16k_audiosetnorm_bs32 \
  --dataset emodb --protocol nested \
  --config eat_base_sr16k_audiosetnorm_bs32 \
  --expected-seeds 3949,42,2026,1024,666 \
  --output /root/songhao/experiments/EAT2EMO/emodb/nested/eat_base_sr16k_audiosetnorm_bs32/summary.json
```

汇总器会严格校验：

- dataset/protocol/config/stage/split；
- fold/seed 网格完整、无重复；
- 每个 seed 的 outer-valid 路径唯一且覆盖预期样本数（EMO-DB 535）；
- `predictions.npz` 与 `metrics.json` 重算一致；
- 标签顺序、类别数、target_length、norm mode/stats 正确。

输出 `summary.json` 和 `summary_folds.csv`。完整的 OOF/均值±标准差、UAR/Macro-F1 口径见 [SER_FINETUNE_GUIDE.md](SER_FINETUNE_GUIDE.md) §6。

> 注意数值尺度：`train.log` / `selection_summary.json` 里的 UAR 是 **0–100**（`tasks/finetuning.py` 乘了 100）；`metrics.json` / `summary.json` 是 **0–1**。混用前先确认文件来源。

---

## 9. 故障排查

| 现象 | 优先检查 |
|---|---|
| `ValueError: numpy.dtype size changed` | `eat2emo` 的 numpy 是否被升到 2.x；应为 1.26.3。旧 `emoat` 不再使用 |
| `Failed to initialize NumPy: _ARRAY_API not found` | 同上；torch 与 numpy 的 ABI 不匹配 |
| `ModuleNotFoundError: No module named 'fairseq'` | 是否用了系统 python 3.13；改用 `PYTHON_BIN=/root/miniconda3/envs/eat2emo/bin/python` |
| `ImportError: huggingface-hub>=0.30.0,<1.0 is required` | 在 `eat2emo` 里 `pip install "huggingface-hub>=0.30,<1.0"` |
| `fairseq_cli/hydra_train.py` 不存在 | `FAIRSEQ_ROOT` 是否指向 `/root/songhao/ProJ/fairseq` |
| `Missing manifest` / `Could not resolve manifest root` | 是否已按 §5.2 生成；TSV 第一行是否为服务器 `SER_DATA_ROOT`；不要用本地旧 manifest |
| `Refusing to overwrite existing stage directory` | 该 fold/stage 已存在；换新的 `SER_OUTPUT_ROOT`，不要删他人结果 |
| `Selection did not produce .../checkpoint_best.pt` | selection 训练失败或被中断；看 `train.log` 尾部；换新目录重跑 |
| 启动时报 `restore_file` / `__fresh_start__.pt` 不存在 | 协议故意用一个不存在的 restore_file 强制 fresh start；先 `--dry-run` 确认；真实启动若失败，记录 Fairseq 版本和完整错误，不要改成 resume 上一阶段 checkpoint |
| `Expected inner-valid UAR for epochs 1..N` | 日志不完整（中断/日志解析失败）；换新目录重跑，不要手改日志 |
| `Selection summary ... pretrained` 不一致 | 本次 `SER_PRETRAINED` 与 selection 时不同；用同一权重，或换新目录 |
| `missing seed/fold results` | 批量没跑完，或 `--expected-seeds` 与实际不符；补齐后再汇总 |
| `CUDA out of memory` | 是否并行跑了多个 fold；确认只有一个训练进程；不要自行改 batch size |
| 音频解码/空音频/NaN 报错 | 数据文件损坏；按路径检查，不要跳过或用其他样本补位 |
| `Permission denied` | 当前用户/目录归属；不要 chown 他人目录；本项目用 root 运行 |
| `git pull` 冲突 / 无法 pull | 服务器当前是 rsync 副本，没有 `.git`；不要执行 `git pull`。要切 Git，按 §3.2 先备份副本再 clone |
| 端口 2222 连不上 | Cactus 隧道是否还在；本机 `lsof -nP -iTCP:2222 -sTCP:LISTEN`；必要时改用 `ssh eaglelab` |
| `screen` 会话丢失 | 容器/作业重启会杀进程；重新开 session，检查 `nvidia-smi` 是否有残留进程 |
| 汇总结果和预期不一致 | 是否混入 smoke/custom config；是否用了旧 manifest；是否重复生成过 manifest |

---

## 10. 收尾

1. 确认每个 fold 的产物完整：

   ```bash
   ls "$SER_OUTPUT_ROOT/emodb/nested/eat_base_sr16k_audiosetnorm_bs32/seed_3949/fold_01/final/results"
   ```

   应包含 `metrics.json`、`predictions.npz`、`paths.txt`、`confusion_matrix.csv`。

2. 确认 `summary.json` 的 `complete=true`，且 `expected_seeds` / `expected_folds` 与实际一致。

3. 需要归档时，只打包自己的结果目录：

   ```bash
   tar -czf "/root/songhao/experiments/EAT2EMO_$(date +%Y%m%d).tar.gz" \
     -C /root/songhao/experiments EAT2EMO
   ```

4. 结束自己的 `screen` 会话：

   ```bash
   screen -ls
   screen -r eat2emo        # 进入后 exit，或 Ctrl-A D 仅 detach
   ```

5. 检查残留 GPU 进程：

   ```bash
   nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
   ```

   只结束本轮自己启动的 PID；不要全局 `pkill`。

6. 不要删除他人的目录或旧实验；只清理自己创建的临时文件。

---

## 11. 待办与未验证边界

已完成（2026-10-19）：

- [x] 本地 commit/push 已完成（SER 协议、IEMOCAP extractor 修复、文档、上游文件清理）；
- [x] 早期代码已 rsync 到 `/root/songhao/ProJ/EAT2EMO`（服务器 60 files / 4.0MB，无 `.git`）；**但副本已过期，训练前需重新同步**；
- [x] 新建 `eat2emo` 环境并完成 §4.2 预检（torch / numpy / sklearn / fairseq / timm / transformers；`pip check` 无 broken requirements）；
- [x] 旧 `emoat` 保留，未删除。

尚未执行：

- [ ] 重新 rsync 服务器代码，使服务器与本地 HEAD 一致；
- [ ] 生成 4 套 manifest（§5.2），核对 counts / `split.json` audits；
- [ ] nested dry-run + smoke（1 fold、1 epoch）；
- [ ] fixed dry-run + smoke（1 fold、1 epoch）；
- [ ] 1 个正式 fold，记录实际 epoch 时间与显存占用；
- [ ] 正式批量（§7.2）；
- [ ] 汇总并核对 `summary.json`；
- [ ] （长期）服务器从 rsync 副本切到 Git（§3.2）。

本轮未验证：

- EAT 微调在 RTX 4090 24GB 上的实际显存占用和单 epoch 时间；
- 批量长跑的稳定性、checkpoint 实际大小和磁盘增长；
- manifest 生成器在 5531 条 IEMOCAP 全量数据上的实际审计结果；
- `checkpoint.restore_file=__fresh_start__.pt` 在服务器 Fairseq 0.12.2 上的实际行为（缺文件是否被容忍）；
- `screen` 在作业容器重启后的存活边界；
- 正式训练/评测尚未开始。

首次正式运行前，建议把 §4.2、§5.2、§6.1 的输出保存到本轮日志目录，作为“本轮已预检”的证据，而不是只凭本文推断环境可用。
