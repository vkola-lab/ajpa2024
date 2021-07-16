#!/bin/bash

DATA="${1-CPTAC/ROIs1024}"
DEVICE="${2-2}"
BATCH_SIZE="${3-6}"
HIDDEN_DIM="${4-64}"
EXP_NAME="${5-Exp03}"
ITERATIONS="${6-100}"

export CUDA_VISIBLE_DEVICES=${DEVICE}
python image_main.py --dataset ${DATA} --fold_idx 0 --batch_size ${BATCH_SIZE} --hidden_dim ${HIDDEN_DIM} --learn_eps --exp_name ${EXP_NAME} --iters_per_epoch ${ITERATIONS}
python image_main.py --dataset ${DATA} --fold_idx 1 --batch_size ${BATCH_SIZE} --hidden_dim ${HIDDEN_DIM} --learn_eps --exp_name ${EXP_NAME} --iters_per_epoch ${ITERATIONS}
python image_main.py --dataset ${DATA} --fold_idx 2 --batch_size ${BATCH_SIZE} --hidden_dim ${HIDDEN_DIM} --learn_eps --exp_name ${EXP_NAME} --iters_per_epoch ${ITERATIONS}
python image_main.py --dataset ${DATA} --fold_idx 3 --batch_size ${BATCH_SIZE} --hidden_dim ${HIDDEN_DIM} --learn_eps --exp_name ${EXP_NAME} --iters_per_epoch ${ITERATIONS}
python image_main.py --dataset ${DATA} --fold_idx 4 --batch_size ${BATCH_SIZE} --hidden_dim ${HIDDEN_DIM} --learn_eps --exp_name ${EXP_NAME} --iters_per_epoch ${ITERATIONS}
python image_main.py --dataset ${DATA} --fold_idx 5 --batch_size ${BATCH_SIZE} --hidden_dim ${HIDDEN_DIM} --learn_eps --exp_name ${EXP_NAME} --iters_per_epoch ${ITERATIONS}
python image_main.py --dataset ${DATA} --fold_idx 6 --batch_size ${BATCH_SIZE} --hidden_dim ${HIDDEN_DIM} --learn_eps --exp_name ${EXP_NAME} --iters_per_epoch ${ITERATIONS}
python image_main.py --dataset ${DATA} --fold_idx 7 --batch_size ${BATCH_SIZE} --hidden_dim ${HIDDEN_DIM} --learn_eps --exp_name ${EXP_NAME} --iters_per_epoch ${ITERATIONS}
python image_main.py --dataset ${DATA} --fold_idx 8 --batch_size ${BATCH_SIZE} --hidden_dim ${HIDDEN_DIM} --learn_eps --exp_name ${EXP_NAME} --iters_per_epoch ${ITERATIONS}
python image_main.py --dataset ${DATA} --fold_idx 9 --batch_size ${BATCH_SIZE} --hidden_dim ${HIDDEN_DIM} --learn_eps --exp_name ${EXP_NAME} --iters_per_epoch ${ITERATIONS}
