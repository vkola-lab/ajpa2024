#!/bin/bash

DEVICE="${1-0}"
RUN_NAME="${2-Graph-Perciever_September-17}"

export CUDA_VISIBLE_DEVICES=${DEVICE}
python plot.py \
--gnn gin \
--num_layer 3 \
--emb_dim 64 \
--jk sum \
--n_folds 5 \
--graph_pooling gmt \
--num_workers 0 \
--dataset cptac cis pcga \
--phase plot \
--n_classes 3 \
--data_config ctranspath_files \
--fdim 768 \
--plot_functions tsne umap pca \
--run_name ${RUN_NAME} \