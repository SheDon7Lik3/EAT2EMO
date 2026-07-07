#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/root/songhao/datasets/checkpoints}"
EAT_PRETRAINED="${EAT_PRETRAINED:-${CHECKPOINT_ROOT}/EAT/checkpoint10.pt}"
ESC50_DATA_ROOT="${ESC50_DATA_ROOT:-/root/songhao/datasets/ser_datasets}"

python fairseq_cli/hydra_train.py -m \
    --config-dir EAT/config \
    --config-name finetuning  \
    common.user_dir=EAT \
    common.log_interval=100 \
    checkpoint.save_dir="${CHECKPOINT_ROOT}/esc50/fold2" \
    checkpoint.restore_file="${CHECKPOINT_ROOT}/esc50/fold2/checkpoint_last.pt" \
    dataset.batch_size=48 \
    criterion.log_keys=['correct'] \
    task.data="${ESC50_DATA_ROOT}/ESC_50/test02" \
    task.esc50_eval=True \
    task.target_length=512 \
    task.roll_aug=true \
    optimization.max_update=4000 \
    optimizer.groups.default.lr_scheduler.warmup_updates=400 \
    model.model_path="${EAT_PRETRAINED}" \
    model.num_classes=50 \
    model.esc50_eval=true \
    model.mixup=0.0 \
    model.target_length=512 \
    model.mask_ratio=0.4 \
    model.label_smoothing=0.1 \
    model.prediction_mode=PredictionMode.CLS_TOKEN \
