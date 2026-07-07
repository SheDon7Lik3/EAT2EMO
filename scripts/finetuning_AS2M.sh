#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/root/songhao/datasets/checkpoints}"
EAT_PRETRAINED="${EAT_PRETRAINED:-${CHECKPOINT_ROOT}/EAT/checkpoint10.pt}"
AS2M_DATA_ROOT="${AS2M_DATA_ROOT:-/root/songhao/datasets/ser_datasets}"

python fairseq_cli/hydra_train.py -m \
    --config-dir EAT/config \
    --config-name finetuning  \
    common.user_dir=EAT \
    checkpoint.save_dir="${CHECKPOINT_ROOT}/finetune_test" \
    checkpoint.restore_file="${CHECKPOINT_ROOT}/finetune_test/checkpoint_last.pt" \
    checkpoint.best_checkpoint_metric=mAP \
    dataset.batch_size=96 \
    task.data="${AS2M_DATA_ROOT}/AS2M" \
    task.h5_format=true \
    task.AS2M_finetune=true \
    task.weights_file="${AS2M_DATA_ROOT}/AS2M/weight_train_all.csv" \
    task.target_length=1024 \
    task.roll_aug=true \
    model.model_path="${EAT_PRETRAINED}" \
    model.num_classes=527 \
    model.mixup=0.8 \
    model.mask_ratio=0.2 \
    model.prediction_mode=PredictionMode.CLS_TOKEN \
