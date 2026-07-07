#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/root/songhao/datasets/checkpoints}"
EAT_PRETRAINED="${EAT_PRETRAINED:-${CHECKPOINT_ROOT}/EAT/checkpoint10.pt}"
AS20K_DATA_ROOT="${AS20K_DATA_ROOT:-/root/songhao/datasets/ser_datasets}"

python fairseq_cli/hydra_train.py -m \
    --config-dir EAT/config \
    --config-name finetuning  \
    common.user_dir=EAT \
    checkpoint.save_dir="${CHECKPOINT_ROOT}/finetune_as20k" \
    checkpoint.restore_file="${CHECKPOINT_ROOT}/finetune_as20k/checkpoint_last.pt" \
    checkpoint.best_checkpoint_metric=mAP \
    dataset.batch_size=48 \
    task.data="${AS20K_DATA_ROOT}/AS20K" \
    task.target_length=1024 \
    task.roll_aug=true \
    optimization.max_update=40000 \
    optimizer.groups.default.lr_scheduler.warmup_updates=4000 \
    model.model_path="${EAT_PRETRAINED}" \
    model.num_classes=527 \
    model.mixup=0.8 \
    model.mask_ratio=0.2 \
    model.prediction_mode=PredictionMode.CLS_TOKEN \
