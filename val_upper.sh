name=cylinder_asym_networks
gpuid=0

CUDA_VISIBLE_DEVICES=${gpuid}  python -u val_cylinder_asym_upper.py \
2>&1 | tee logs_dir/${name}_logs_tee.txt