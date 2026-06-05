#!/bin/bash
# Validate pretrained ImageNet models on GPU 3

cd /mnt/disk1/st/sdtv1/Spike-Driven-Transformer

source ~/miniconda3/etc/profile.d/conda.sh
conda activate spikeTransformer

export CUDA_VISIBLE_DEVICES=3

echo "=========================================="
echo "Testing 8_384 model (expected: 72.28% top-1)"
echo "=========================================="
python firing_num.py \
    -c ./conf/imagenet/8_384_300E_t4.yml \
    --model sdt \
    --spike-mode lif \
    --resume ./pretrained/8_384.pth \
    --no-resume-opt

echo ""
echo "=========================================="
echo "Testing 8_768 model (expected: 77.07% top-1)"
echo "=========================================="
python firing_num.py \
    -c ./conf/imagenet/8_768_300E_t4.yml \
    --model sdt \
    --spike-mode lif \
    --resume ./pretrained/8_768.pth \
    --no-resume-opt

echo ""
echo "Validation complete!"
