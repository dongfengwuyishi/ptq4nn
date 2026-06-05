#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29001 firing_num.py -c ./conf/cifar100/4_384_300E_t4.yml --model sdt --spike-mode lif --resume output/train/20230224-154007-ms_spikeformer-data-cifar100-t-4-spike-lif-attn-hydra_direct_lif/model_best.pth.tar --no-resume-opt

CUDA_VISIBLE_DEVICES=3 /usr/bin/python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 29001 firing_num.py \
    -c ./conf/cifar10/2_256_300E_t4.yml \
    --model sdt \
    --spike-mode lif \
    --resume output/train/20260115-214127-sdt-data-cifar10-t-4-spike-lif/model_best.pth.tar \
    --no-resume-opt