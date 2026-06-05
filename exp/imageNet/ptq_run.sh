#!/bin/bash
# PTQ 运行脚本 - imageNet
# 在项目根目录执行: ./exp/imageNet/ptq_run.sh

cd "$(dirname "$0")/../.."
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate spikeTransformer

CONFIG="exp/imageNet/config.yml"
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config not found: $CONFIG"
    exit 1
fi
echo "Config: $CONFIG"
python ptq/main.py --config "$CONFIG"
