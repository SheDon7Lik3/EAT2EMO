# 环境配置

本项目建议使用 Python 3.10，并配合 CUDA 12.1 的 PyTorch 安装包。

## 1. 创建 conda 环境

```bash
conda create -n emoat python=3.10 -y
conda activate emoat
```

## 2. 安装 PyTorch

```bash
pip install \
  torch==2.1.2 \
  torchvision==0.16.2 \
  torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu121 \
  --timeout 1000 \
  --retries 10
```

## 3. 固定 pip 和基础构建工具

```bash
python -m pip install pip==24.0
pip install \
  setuptools==69.5.1 \
  numpy==1.26.3 \
  wheel \
  cython
```

## 4. 安装 Fairseq

如果你要做预训练、微调或 SER 评估，需要安装 Fairseq。

```bash
git clone https://github.com/pytorch/fairseq
cd ~/songhao/ProJ/fairseq
pip install --editable ./ --no-build-isolation
```

如果你的 Fairseq 不在这个路径下，把 `cd` 后面的目录改成你自己的即可。

## 5. 安装 EAT 项目依赖

在本仓库根目录下执行：

```bash
cd /path/to/EAT2EMO
pip install timm==0.9.12 --no-deps
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 6. 推荐的环境变量

这些变量能让服务器上的脚本更方便复用：

```bash
export FAIRSEQ_ROOT=/path/to/fairseq
export SER_DATA_ROOT=/root/songhao/datasets/ser_datasets
export CHECKPOINT_ROOT=/root/songhao/datasets/checkpoints
```

## 7. 快速验证

```bash
python -c "import torch; print(torch.__version__)"
python -c "import fairseq; print('fairseq ok')"
python -c "import timm; print(timm.__version__)"
```

## 说明

- 请保持 `pip` 版本低于 `24.1`，本项目建议使用 `pip==24.0`。
- 仓库里的 `requirements.txt` 已经包含了特征提取和推理所需的运行依赖。
- 如果你只需要 Hugging Face 推理或特征提取，Fairseq 不是必须的。
