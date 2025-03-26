#!/bin/bash
export CURRENT_DIR=$(pwd)
echo "CURRENT_DIR=${CURRENT_DIR}"
export WANDB_MODE=online
echo "WANDB_MODE=${WANDB_MODE}"
export HYDRA_FULL_ERROR=1
echo "HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR}"
export DISABLE_MUJOCO=1 # if this env var exists and is not 0, it will "disable" mujoco ...
# ... meaning you can import gym environment that don't use it without getting an error
echo "DISABLE_MUJOCO=${DISABLE_MUJOCO}"
python dino_wm/custom_plan.py