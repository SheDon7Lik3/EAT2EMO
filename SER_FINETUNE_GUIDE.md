# SER 微调指南

本文档说明如何在本项目中进行 Speech Emotion Recognition, SER 微调。当前默认使用服务器上的数据路径与 checkpoint 路径，适合在本地开发后同步到服务器执行。

## 1. 目录约定

默认约定如下：

- 代码仓库：`/root/songhao/ProJ/EAT2EMO`
- SER 数据集根目录：`/root/songhao/datasets/ser_datasets`
- checkpoint 根目录：`/root/songhao/datasets/checkpoints`
- Fairseq 源码：`/root/songhao/ProJ/fairseq`

SER 原始数据放在：

- `/root/songhao/datasets/ser_datasets/ravdess-speech`
- `/root/songhao/datasets/ser_datasets/emodb`
- `/root/songhao/datasets/ser_datasets/iemocap`

SER 划分后的 manifests 会生成到：

- `/root/songhao/datasets/ser_datasets/manifests/ravdess`
- `/root/songhao/datasets/ser_datasets/manifests/emodb`
- `/root/songhao/datasets/ser_datasets/manifests/iemocap`

每个数据集目录下面会有多个 fold，例如：

- `/root/songhao/datasets/ser_datasets/manifests/ravdess/fold_01`
- `/root/songhao/datasets/ser_datasets/manifests/ravdess/fold_02`

每个 fold 下会包含：

- `train.tsv`
- `eval.tsv`
- `test.tsv`
- `train.lbl`
- `eval.lbl`
- `test.lbl`
- `label_descriptors.csv`
- `summary.json`

## 2. 环境准备

先确认相关环境变量：

```bash
export FAIRSEQ_ROOT=/root/songhao/ProJ/fairseq
export SER_DATA_ROOT=/root/songhao/datasets/ser_datasets
export CHECKPOINT_ROOT=/root/songhao/datasets/checkpoints
```

如果第一次使用，请先确认以下文件存在：

```bash
test -f /root/songhao/ProJ/fairseq/fairseq_cli/hydra_train.py
test -d /root/songhao/datasets/ser_datasets/ravdess-speech
test -d /root/songhao/datasets/ser_datasets/emodb
test -d /root/songhao/datasets/ser_datasets/iemocap
test -f /root/songhao/datasets/checkpoints/EAT-base_epoch30_pt.pt
```

## 3. 生成 SER manifests

当原始数据放好后，先生成 speaker-independent manifests：

```bash
cd /root/songhao/ProJ/EAT2EMO
python tools/prepare_ser_manifests.py \
  --datasets-root /root/songhao/datasets/ser_datasets \
  --output /root/songhao/datasets/ser_datasets/manifests
```

生成完成后，可以检查任意一个 fold：

```bash
ls /root/songhao/datasets/ser_datasets/manifests/ravdess/fold_01
head -n 3 /root/songhao/datasets/ser_datasets/manifests/ravdess/fold_01/eval.tsv
```

## 4. 训练

SER 微调脚本为：

```bash
bash scripts/finetuning_SER.sh ravdess fold_01
```

默认会使用：

- 数据集：`ravdess`
- fold：`fold_01`
- 预训练模型：`/root/songhao/datasets/checkpoints/EAT-base_epoch30_pt.pt`
- 输出目录：`/root/songhao/datasets/checkpoints/ser/ravdess/fold_01`

如果你想指定其他预训练 checkpoint，可以作为第三个参数传入：

```bash
bash scripts/finetuning_SER.sh ravdess fold_01 /path/to/your_checkpoint.pt
```

如果你想指定输出目录，可以作为第四个参数传入：

```bash
bash scripts/finetuning_SER.sh ravdess fold_01 /path/to/your_checkpoint.pt /path/to/output_dir
```

### 支持的数据集

目前脚本支持：

- `ravdess`
- `emodb`
- `iemocap`

## 5. 评估

训练结束后，单独跑测试集评估：

```bash
bash scripts/eval_SER.sh ravdess fold_01
```

默认会读取：

- checkpoint：`/root/songhao/datasets/checkpoints/ser/ravdess/fold_01/checkpoint_best.pt`
- 输出目录：`/root/songhao/datasets/checkpoints/ser/ravdess/fold_01/test`

如果想手动指定 checkpoint，可以作为第三个参数传入：

```bash
bash scripts/eval_SER.sh ravdess fold_01 /path/to/checkpoint_best.pt
```

如果想指定测试输出目录，可以作为第四个参数传入：

```bash
bash scripts/eval_SER.sh ravdess fold_01 /path/to/checkpoint_best.pt /path/to/test_output
```

测试结果会写入：

- `metrics.json`
- `confusion_matrix.csv`

## 6. 汇总交叉验证结果

当一个数据集的所有 fold 都跑完后，可以汇总结果：

```bash
python evaluation/summarize_ser.py /root/songhao/datasets/checkpoints/ser/ravdess
```

汇总脚本会读取所有：

- `fold_*/test/metrics.json`

并输出各指标的 mean/std。

## 7. 常见问题

### 7.1 manifest root 找不到

如果训练时报错类似：

```text
Could not resolve manifest root ...
```

通常说明 `train.tsv` / `eval.tsv` 第一行写的 root 不对。  
解决方式是重新生成 manifests：

```bash
python tools/prepare_ser_manifests.py \
  --datasets-root /root/songhao/datasets/ser_datasets \
  --output /root/songhao/datasets/ser_datasets/manifests
```

### 7.2 找不到 Fairseq

如果报错找不到 `fairseq_cli/hydra_train.py`，说明 `FAIRSEQ_ROOT` 没有指对。  
请确认：

```bash
ls /root/songhao/ProJ/fairseq/fairseq_cli/hydra_train.py
```

### 7.3 checkpoint 不存在

默认微调脚本会读取：

```bash
/root/songhao/datasets/checkpoints/EAT-base_epoch30_pt.pt
```

如果没有这个文件，请换成你自己的预训练 checkpoint。

## 8. 推荐的运行顺序

建议按下面顺序执行：

1. 配置环境变量
2. 生成 SER manifests
3. 训练一个 fold
4. 跑该 fold 的测试集
5. 汇总所有 fold

例如：

```bash
export FAIRSEQ_ROOT=/root/songhao/ProJ/fairseq
export SER_DATA_ROOT=/root/songhao/datasets/ser_datasets
export CHECKPOINT_ROOT=/root/songhao/datasets/checkpoints

cd /root/songhao/ProJ/EAT2EMO
python tools/prepare_ser_manifests.py \
  --datasets-root /root/songhao/datasets/ser_datasets \
  --output /root/songhao/datasets/ser_datasets/manifests

bash scripts/finetuning_SER.sh ravdess fold_01
bash scripts/eval_SER.sh ravdess fold_01
python evaluation/summarize_ser.py /root/songhao/datasets/checkpoints/ser/ravdess
```

