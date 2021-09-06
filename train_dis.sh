name=cylinder_asym_networks
gpuid=0,1

CUDA_VISIBLE_DEVICES=${gpuid}  python -u -m torch.distributed.launch --nproc_per_node=2 train_cylinder_asym_dis.py \
2>&1 | tee logs_dir/${name}_logs_tee.txt