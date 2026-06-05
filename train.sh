#!/bin/bash
export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH

# CUDA_VISIBLE_DEVICES=3 /usr/bin/python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 29516 train.py -c conf/cifar100/2_256_300E_t4.yml --model sdt --spike-mode lif

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate spikeTransformer
# export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH
# CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=7 --master_port 29502 train.py -c conf/imagenet/6_512_300E_t4.yml --model sdt --spike-mode lif

CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port 29516 train.py -c conf/imagenet/8_768_300E_t4_2gpu.yml --model sdt --spike-mode lif