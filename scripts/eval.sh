#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/root/songhao/datasets/checkpoints}"
AS20K_DATA_ROOT="${AS20K_DATA_ROOT:-/root/songhao/datasets/ser_datasets}"

python EAT/evaluation/eval.py  \
    --label_file='EAT/inference/labels.csv' \
    --eval_dir="${AS20K_DATA_ROOT}/AS20K" \
    --model_dir='EAT' \
    --checkpoint_dir="${CHECKPOINT_ROOT}/EATs/finetuning/as20k_epoch30/checkpoint_last.pt" \
    --target_length=1024 \
    --device='cuda' \
    --batch_size=32

# For optimal performance, 1024 is recommended for 10-second audio clips. (128 for 1-second)
# However, you should adjust the target_length parameter based on the duration and characteristics of your specific audio inputs.
# EAT-finetuned could make evaluation well even given truncated audio clips.
