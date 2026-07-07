#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/root/songhao/datasets/checkpoints}"
EAT_PRETRAINED="${EAT_PRETRAINED:-${CHECKPOINT_ROOT}/EAT/checkpoint10.pt}"
SPCV2_DATA_ROOT="${SPCV2_DATA_ROOT:-/root/songhao/datasets/ser_datasets}"

python fairseq_cli/hydra_train.py -m \
    --config-dir EAT/config \
    --config-name finetuning  \
    common.user_dir=EAT \
    common.seed=42 \
    checkpoint.save_dir="${CHECKPOINT_ROOT}/spcv2" \
    checkpoint.restore_file="${CHECKPOINT_ROOT}/spcv2/checkpoint_last.pt" \
    dataset.batch_size=256 \
    criterion.log_keys=['correct'] \
    task.data="${SPCV2_DATA_ROOT}/SPC_2" \
    task.spcv2_eval=True \
    task.target_length=128 \
    task.noise=true \
    task.roll_aug=true \
    optimization.lr=[0.0002] \
    optimizer.groups.default.lr_float=0.0002 \
    optimization.max_update=40000 \
    optimizer.groups.default.lr_scheduler.warmup_updates=4000 \
    model.model_path="${EAT_PRETRAINED}" \
    model.num_classes=35 \
    model.spcv2_eval=true \
    model.mixup=0.8 \
    model.target_length=128 \
    model.mask_ratio=0.2 \
    model.label_smoothing=0.1 \
    model.prediction_mode=PredictionMode.CLS_TOKEN \
