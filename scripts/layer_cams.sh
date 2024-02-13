
#!/bin/bash
DEVICE="${1-0}"
FOLD_IDX="${2-0}"
RUN_NAME="${3-Graph-Perciever_September-17}"

export CUDA_VISIBLE_DEVICES=${DEVICE}
python layer_cams.py \
--gnn gin \
--num_layer 3 \
--emb_dim 64 \
--jk sum \
--graph_pooling gmt \
--batch_size 1 \
--num_workers 4 \
--dataset cptac \
--phase cams \
--n_classes 3 \
--data_config ctranspath_files \
--fdim 768 \
--run_name ${RUN_NAME} \
--fold_idx ${FOLD_IDX} 
 
