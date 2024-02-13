
#!/bin/bash

DEVICE="${1-0}"
CONFIG_FILE="${2-configs/test_config.yaml}"

export CUDA_VISIBLE_DEVICES=${DEVICE}
python test.py --config_file ${CONFIG_FILE}