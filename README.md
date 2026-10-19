# EAT2EMO

Internal project: EAT audio encoder + Speech Emotion Recognition (SER) fine-tuning.

**Documentation**

| 文档 | 负责内容 |
|---|---|
| [ENVIRONMENT.md](./ENVIRONMENT.md) | 环境安装、版本固定、当前 `eat2emo` 服务器实况 |
| [SER_FINETUNE_GUIDE.md](./SER_FINETUNE_GUIDE.md) | SER 协议、单 fold/批量命令、指标口径 |
| [GPU-RUNBOOK.md](./GPU-RUNBOOK.md) | GPU 服务器连接、rsync 部署、运行、smoke 与排障 |

## What this repo is

- 基于 [EAT](https://github.com/cwx-worst-one/EAT) 的音频自监督编码器，使用 EAT-base epoch30 预训练权重。
- 在本仓库中增加了 SER 微调协议：EMO-DB、IEMOCAP-5531、RAVDESS-Speech、RAVDESS-Song，支持 nested / fixed 两种协议。
- 训练、评测、汇总的入口在 `scripts/`、`tools/`、`evaluation/`；具体约定见 SER 指南。
- AudioSet 预训练、AS-2M/AS-20K/ESC-50/SPC-2 微调和 AudioSet 推理仍以上游 EAT 仓库为准；本仓库不再保留这些上游启动脚本和配置。

## Requirements and Installation

推荐 Python 3.10 + CUDA 12.1 PyTorch。完整安装步骤、固定版本和当前服务器环境见 [ENVIRONMENT.md](./ENVIRONMENT.md)。

服务器上的代码同步、环境预检和运行入口见 [GPU-RUNBOOK.md](./GPU-RUNBOOK.md)。

## Feature Extraction

`feature_extract/feature_extract.py` 从冻结的 EAT 编码器提取特征并保存为 `.npy`，支持 Fairseq checkpoint 或 Hugging Face 模型。

Hugging Face checkpoint、utterance-level CLS 特征示例（确保 `PYTHON_BIN` 指向 `eat2emo` 环境，见 [ENVIRONMENT.md](./ENVIRONMENT.md)）：

```bash
$PYTHON_BIN feature_extract/feature_extract.py \
  --source_file feature_extract/test.wav \
  --target_file feature_extract/test.npy \
  --model_dir . \
  --checkpoint_dir worstchan/EAT-base_epoch30_finetune_AS2M \
  --granularity utterance \
  --target_length 1024 \
  --mode finetune \
  --framework huggingface
```

使用 Fairseq checkpoint 时改为 `--framework fairseq --checkpoint_dir /path/to/checkpoint.pt`，并保留 `--model_dir .`。完整参数见 [feature_extract/feature_extract.py](./feature_extract/feature_extract.py)。

**Notes**

- 提取的是冻结 EAT 编码器的特征，帧级特征约 50 Hz。
- `--granularity`：`all`（帧级，含 CLS）、`frame`（帧级，不含 CLS）、`utterance`（只保留 CLS，分类任务通常优先）。
- `target_length`：10 秒音频用 `1024`，5 秒用 `512`；EAT 预训练使用 10 秒 AudioSet 片段，建议尽量 pad/trim 到 10 秒。
- CNN 编码器使用 16×16 patch、stride 16，因此 `target_length` 必须是 **16 的倍数**。
- 预训练 checkpoint 适合再微调；微调后的 checkpoint 通常更适合直接提特征。

## Speech Emotion Recognition

SER 使用 EAT 编码器的 CLS token 做单标签分类，数据划分为 speaker/session-independent。支持：

| 数据集 | 类别数 | fold | 默认 target_length |
|---|---:|---|---:|
| EMO-DB | 7 | 1–10 | 512 |
| IEMOCAP-5531 | 4 | 1–5 | 1024 |
| RAVDESS-Speech | 8 | 0–4 | 512 |
| RAVDESS-Song | 6 | 0–4 | 640 |

环境变量、manifest 生成、单 fold 和批量命令见 [SER_FINETUNE_GUIDE.md](./SER_FINETUNE_GUIDE.md)。最常用的入口：

```bash
# 生成/审计 manifests（首次）
$PYTHON_BIN tools/prepare_ser_manifests.py \
  --datasets-root "$SER_DATA_ROOT" \
  --manifest-root "$SER_DATA_ROOT" \
  --output "$SER_DATA_ROOT/manifests"

# 单 fold：nested / fixed
bash scripts/finetuning_SER.sh --dataset emodb --fold 1 --seed 3949 --protocol nested
bash scripts/finetuning_SER.sh --dataset emodb --fold 1 --seed 3949 --protocol fixed

# 全部数据集 × 5 seeds（3949,42,2026,1024,666），并自动汇总
bash scripts/run_all_SER.sh --protocol nested
bash scripts/run_all_SER.sh --protocol fixed --fixed-epochs 60
```

服务器上先按 [GPU-RUNBOOK.md](./GPU-RUNBOOK.md) §4.1 导出环境变量；不要用系统默认 `python`。

## Upstream EAT

- 论文：[EAT: Self-Supervised Pre-Training with Efficient Audio Transformer](https://arxiv.org/abs/2401.03497)
- 上游代码：[cwx-worst-one/EAT](https://github.com/cwx-worst-one/EAT)
- Hugging Face：[EAT collection](https://huggingface.co/collections/worstchan/eat-6815b4f1034f5214f9063948)
- 上游预训练/微调、AudioSet 数据准备、AudioSet 推理与 mAP 评测，请直接使用上游仓库；本仓库聚焦 SER，已移除这些上游启动脚本和配置。

常用 checkpoint：

| Version | Google Drive | Hugging Face |
|---|---|---|
| EAT-base epoch30 pretrained | [Link](https://drive.google.com/file/d/19hfzLgHCkyqTOYmHt8dqVa9nm-weBq4f/view?usp=sharing) | [Link](https://huggingface.co/worstchan/EAT-base_epoch30_pretrain) |
| EAT-base epoch30 AS-2M | [Link](https://drive.google.com/file/d/1aCYiQmoZv_Gh1FxnR-CCWpNAp6DIJzn6/view?usp=sharing) | [Link](https://huggingface.co/worstchan/EAT-base_epoch30_finetune_AS2M) |
| EAT-large epoch20 pretrained | [Link](https://drive.google.com/file/d/1PEgriRvHsqrtLzlA478VemX7Q0ZGl889/view?usp=sharing) | [Link](https://huggingface.co/worstchan/EAT-large_epoch20_pretrain) |
| EAT-large epoch20 AS-2M | [Link](https://drive.google.com/file/d/1b_f_nQAdjM1B6u72OFUtFiUu-4yM2shd/view?usp=sharing) | [Link](https://huggingface.co/worstchan/EAT-large_epoch20_finetune_AS2M) |

上游 AudioSet 性能（mAP %）：

| Model | Backbone | Params | AS-20K | AS-2M |
|:-:|:-:|:-:|:-:|:-:|
| EAT-base (10 epochs) | ViT-B | 88M | 40.3 | 48.6 |
| EAT-base (30 epochs) | ViT-B | 88M | 41.3 | 48.9 |
| EAT-large (20 epochs) | ViT-L | 309M | 42.0 | 49.5 |

上游实现基于 [Audio-MAE](https://github.com/facebookresearch/AudioMAE) 和 [data2vec](https://github.com/facebookresearch/fairseq/tree/main/examples/data2vec)。

## Citation

```bibtex
@inproceedings{chen2024eat,
  title={{EAT}: Self-Supervised Pre-Training with Efficient Audio Transformer},
  author={Chen, Wenxi and Liang, Yuzhe and Ma, Ziyang and Zheng, Zhisheng and Chen, Xie},
  booktitle={IJCAI},
  year={2024}
}
```

## Acknowledgement

- EAT: Shanghai Jiao Tong University / Peng Cheng Laboratory.
- 上游代码基于 [Audio-MAE](https://github.com/facebookresearch/AudioMAE) 和 [data2vec](https://github.com/facebookresearch/fairseq/tree/main/examples/data2vec)。

## License

MIT License，见 [LICENSE](./LICENSE)。
