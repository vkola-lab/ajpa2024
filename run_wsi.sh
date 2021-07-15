#!/bin/bash

DATA="${1-CPTAC}"
DEVICE="${2-2}"
BATCH_SIZE="${3-6}"
HIDDEN_DIM="${4-64}"

export CUDA_VISIBLE_DEVICES=${DEVICE}
python image_main.py --dataset ${DATA} --fold_idx 0 --batch_size ${BATCH_SIZE} --hidden_dim ${HIDDEN_DIM} --learn_eps
python image_main.py --dataset ${DATA} --fold_idx 1 --batch_size ${BATCH_SIZE} --hidden_dim ${HIDDEN_DIM} --learn_eps
python image_main.py --dataset ${DATA} --fold_idx 2 --batch_size ${BATCH_SIZE} --hidden_dim ${HIDDEN_DIM} --learn_eps
python image_main.py --dataset ${DATA} --fold_idx 3 --batch_size ${BATCH_SIZE} --hidden_dim ${HIDDEN_DIM} --learn_eps
python image_main.py --dataset ${DATA} --fold_idx 4 --batch_size ${BATCH_SIZE} --hidden_dim ${HIDDEN_DIM} --learn_eps
python image_main.py --dataset ${DATA} --fold_idx 5 --batch_size ${BATCH_SIZE} --hidden_dim ${HIDDEN_DIM} --learn_eps
python image_main.py --dataset ${DATA} --fold_idx 6 --batch_size ${BATCH_SIZE} --hidden_dim ${HIDDEN_DIM} --learn_eps
python image_main.py --dataset ${DATA} --fold_idx 7 --batch_size ${BATCH_SIZE} --hidden_dim ${HIDDEN_DIM} --learn_eps
python image_main.py --dataset ${DATA} --fold_idx 8 --batch_size ${BATCH_SIZE} --hidden_dim ${HIDDEN_DIM} --learn_eps
python image_main.py --dataset ${DATA} --fold_idx 9 --batch_size ${BATCH_SIZE} --hidden_dim ${HIDDEN_DIM} --learn_eps
