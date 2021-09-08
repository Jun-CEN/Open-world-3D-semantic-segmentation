name=cylinder_asym_networks_nusc
gpuid=1

CUDA_VISIBLE_DEVICES=${gpuid}  python -u train_cylinder_asym_nuscenes_naive.py \
2>&1 | tee logs_dir/${name}_logs_tee.txt