#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/root/songhao/datasets/checkpoints}"
AS2M_DATA_ROOT="${AS2M_DATA_ROOT:-/root/songhao/datasets/ser_datasets}"

python fairseq_cli/hydra_train.py -m \
    --config-dir EAT/config \
    --config-name pretraining_AS2M \
    common.user_dir=EAT \
    checkpoint.save_dir="${CHECKPOINT_ROOT}/test" \
    checkpoint.restore_file="${CHECKPOINT_ROOT}/test/checkpoint_last.pt" \
    distributed_training.distributed_world_size=4 \
    dataset.batch_size=12 \
    task.data="${AS2M_DATA_ROOT}/unbalanced_train" \
    task.h5_format=True \
