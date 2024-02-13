#!/bin/bash

DEVICE="${1-0}"
FOLD="${2-0}"
CONFIG_FILE="${3-configs/config.yaml}"

export CUDA_VISIBLE_DEVICES=${DEVICE}

python main_pyg.py --config_file ${CONFIG_FILE} --fold_idx ${FOLD}
