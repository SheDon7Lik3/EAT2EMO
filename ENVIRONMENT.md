# 环境配置

本项目建议使用 **Python 3.10 + CUDA 12.1 的 PyTorch**。服务器当前使用 conda 环境 **`eat2emo`**（2026-10-19 已验证）；旧环境 `emoat` 保留作回退，不再作为本项目默认环境。

本文分两部分：

- **A. 从零安装**：新机器/新环境的重建步骤；
- **B. 当前服务器实况**：已经装好的版本和验证结果。

本文是**环境安装与版本信息的唯一来源**。SER 协议与指标见 [SER_FINETUNE_GUIDE.md](SER_FINETUNE_GUIDE.md)；服务器连接、代码同步与运行操作见 [GPU-RUNBOOK.md](GPU-RUNBOOK.md)。

---

## A. 从零安装

### 1. 创建 conda 环境

```bash
conda create -n eat2emo python=3.10 -y
conda activate eat2emo

which python
python -V
```

预期 Python 为 3.10.x，路径在 `/root/miniconda3/envs/eat2emo/bin/python`。

### 2. 安装 PyTorch（CU121）

```bash
python -m pip install \
  torch==2.1.2 \
  torchvision==0.16.2 \
  torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu121 \
  --timeout 1000 \
  --retries 10
```

### 3. 固定 pip / setuptools / numpy / Cython

```bash
python -m pip install pip==24.0
python -m pip install setuptools==69.5.1 "cython<3" numpy==1.26.3 wheel
```

说明：

- `pip` 必须保持 `< 24.1`，本文按 `24.0` 固定；看到“可升级到 26.x”的提示不要升。
- `numpy` 必须是 `1.26.3`；`numpy 2.x` 会让 scikit-learn 1.3.2 的预编译扩展报 `numpy.dtype size changed`。
- `Cython` 建议 `<3`（实际使用 0.29.37），Fairseq 0.12.2 的旧扩展用它构建更稳。

### 4. 安装 Fairseq

服务器已有 Fairseq 源码 `/root/songhao/ProJ/fairseq`，直接使用：

```bash
cd /root/songhao/ProJ/fairseq
git rev-parse --short HEAD          # 预期 3d262bb2

python -m pip install --editable ./ --no-build-isolation
```

如果机器上没有源码，再 clone：

```bash
git clone https://github.com/pytorch/fairseq /root/songhao/ProJ/fairseq
cd /root/songhao/ProJ/fairseq
git checkout 3d262bb2
python -m pip install --editable ./ --no-build-isolation
```

Fairseq 安装会编译 Cython 扩展，可能需要几分钟；建议在 `screen` 里执行。

### 5. 安装 EAT 依赖

服务器代码目录为 `/root/songhao/ProJ/EAT2EMO`。如果代码还没有：

```bash
cd /root/songhao/ProJ
[ -d EAT2EMO ] || git clone https://github.com/SheDon7Lik3/EAT2EMO.git EAT2EMO
cd /root/songhao/ProJ/EAT2EMO
```

安装依赖：

```bash
python -m pip install timm==0.9.12 --no-deps

python -m pip install -r requirements.txt \
  -i https://mirrors.aliyun.com/pypi/simple

# transformers 4.51.3 需要 huggingface-hub < 1.0
python -m pip install "huggingface-hub>=0.30.0,<1.0"
```

说明：

- `fairseq` 安装时可能先把 `scikit-learn` 拉到新版本；`requirements.txt` 会再把它降回 `1.3.2`，这是预期行为。
- `huggingface-hub` 主要影响 Hugging Face 推理/特征提取；Fairseq SER 训练本身不依赖 `transformers`。
- 镜像不可用时，去掉 `-i ...` 用默认 PyPI 重试。

### 6. 推荐环境变量

```bash
export REPO=/root/songhao/ProJ/EAT2EMO
export FAIRSEQ_ROOT=/root/songhao/ProJ/fairseq
export SER_DATA_ROOT=/root/songhao/datasets/SER_DATASET
export SER_OUTPUT_ROOT=/root/songhao/experiments/EAT2EMO
export SER_PRETRAINED=/root/songhao/ProJ/resources/EAT_Weight/EAT-base_epoch30_pt.pt

export PYTHON_BIN=/root/miniconda3/envs/eat2emo/bin/python
export PYTHONDONTWRITEBYTECODE=1
```

注意：

- **`PYTHON_BIN` 必须显式设置**；脚本默认退回 PATH 里的 `python`，而服务器系统 python 是 3.13.13，不适合本项目。
- SER 流程不使用 `CHECKPOINT_ROOT`。它是旧文档/旧非 SER 脚本的变量，服务器上 `/root/songhao/datasets/checkpoints` 也不存在。
- 数据集和权重不要放进 Git；它们已经在服务器上。

### 7. 快速验证

```bash
export PYTHON_BIN=/root/miniconda3/envs/eat2emo/bin/python
export PYTHONDONTWRITEBYTECODE=1

$PYTHON_BIN -V
$PYTHON_BIN -m pip -V
$PYTHON_BIN -m pip check

$PYTHON_BIN - <<'PY'
import numpy, sklearn, torch, torchaudio, timm, fairseq, transformers

print("numpy       ", numpy.__version__)
print("sklearn     ", sklearn.__version__)
print("torch       ", torch.__version__)
print("cuda        ", torch.cuda.is_available())
print("device      ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("torchaudio  ", torchaudio.__version__)
print("timm        ", timm.__version__)
print("fairseq     ", fairseq.__version__)
print("fairseq path", fairseq.__file__)
print("transformers", transformers.__version__)
PY

ls -l /root/songhao/ProJ/fairseq/fairseq_cli/hydra_train.py
nvidia-smi
```

预期：

| 检查 | 值 |
|---|---|
| Python | 3.10.x |
| pip | 24.0 |
| numpy | 1.26.3 |
| scikit-learn | 1.3.2 |
| torch | 2.1.2+cu121，`cuda True` |
| fairseq | 0.12.2，指向 `/root/songhao/ProJ/fairseq/fairseq/__init__.py` |
| timm | 0.9.12 |
| transformers | 4.51.3 |
| `pip check` | `No broken requirements found` |

### 8. 常见环境问题

| 现象 | 处理 |
|---|---|
| `ValueError: numpy.dtype size changed` | numpy 被升到 2.x；`python -m pip install numpy==1.26.3` |
| `Failed to initialize NumPy: _ARRAY_API not found` | 同上；torch/sklearn 与 numpy ABI 不匹配 |
| `ImportError: huggingface-hub>=0.30.0,<1.0 is required` | `python -m pip install "huggingface-hub>=0.30,<1.0"` |
| Fairseq 编译 Cython 失败 | 确认 `cython<3`、`setuptools==69.5.1`；必要时 `cython==0.29.36` 后重试 |
| `ModuleNotFoundError: No module named 'fairseq'` | 检查是否用了 `eat2emo` 的绝对路径；确认 `FAIRSEQ_ROOT`/editable 安装 |
| `pip` 提示升级到 26.x | 不要升；保持 `pip==24.0` |
| `pip check` 出现 omegaconf 的 deprecation 提示 | 只是提示，不是 broken requirement；可忽略 |

---

## B. 当前服务器实况（2026-10-19）

| 项目 | 实际值 |
|---|---|
| conda 环境 | `/root/miniconda3/envs/eat2emo`（旧 `emoat` 保留作回退） |
| Python | 3.10.20 |
| pip | 24.0 |
| setuptools | 69.5.1 |
| Cython | 0.29.37 |
| numpy | 1.26.3 |
| scikit-learn | 1.3.2 |
| torch / torchvision / torchaudio | 2.1.2+cu121 / 0.16.2+cu121 / 2.1.2+cu121 |
| CUDA | `torch.cuda.is_available() == True`，NVIDIA RTX 4090 24GB |
| fairseq | 0.12.2，editable 指向 `/root/songhao/ProJ/fairseq` |
| hydra-core / omegaconf | 1.0.7 / 2.0.6 |
| timm | 0.9.12 |
| transformers / tokenizers | 4.51.3 / 0.21.4 |
| soundfile / pyarrow / h5py | 0.12.1 / 15.0.0 / 3.10.0 |
| scipy | 1.15.3 |
| `pip check` | `No broken requirements found`（omegaconf deprecation 警告除外） |
| 代码目录 | `/root/songhao/ProJ/EAT2EMO`（rsync 副本，无 `.git`，约 4.0MB） |
| 数据根 | `/root/songhao/datasets/SER_DATASET`（535 / 5531 / 1440 / 1012） |
| 预训练权重 | `/root/songhao/ProJ/resources/EAT_Weight/EAT-base_epoch30_pt.pt`（1.4G，SHA256 `91228f59...03816`） |
| manifest / 实验输出 | 尚未生成 |
| 系统默认 python | `/root/miniconda3/bin/python` 3.13.13，**不要使用** |

构建与验证记录见 [GPU-RUNBOOK.md](GPU-RUNBOOK.md) 的 §4。

> 本文于 2026-10-19 在本地更新；服务器上的 `ENVIRONMENT.md` 仍是同步时的旧版，本轮未再修改服务器。
