#!/bin/bash
# PTQ 运行脚本
# 用法:
#   ./ptq_run.sh                    # 使用默认 config.yml
#   ./ptq_run.sh my_config.yml      # 使用指定配置

cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate spikeTransformer

CONFIG="${1:-exp/config.yml}"

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config not found: $CONFIG"
    exit 1
fi

echo "Config: $CONFIG"
python ptq/main.py --config "$CONFIG"
