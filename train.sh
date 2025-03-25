#!/bin/bash
export DATASET_DIR=$(pwd)/dataset_dino
echo "DATASET_DIR=${DATASET_DIR}"
export WANDB_MODE=online # set to disabled if issue with wandb
echo "WANDB_MODE=${WANDB_MODE}"
export HYDRA_FULL_ERROR=1
echo "HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR}"
python dino_wm/train.py