#!/bin/bash
# PTQ 运行脚本 - CIFAR10
# 在项目根目录执行: ./exp/cifar10/ptq_run.sh

cd "$(dirname "$0")/../.."
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate spikeTransformer

CONFIG="exp/cifar10/config.yml"
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config not found: $CONFIG"
    exit 1
fi
echo "Config: $CONFIG"
python ptq/main.py --config "$CONFIG"
