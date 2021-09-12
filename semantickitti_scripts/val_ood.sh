name=cylinder_asym_networks
gpuid=1

CUDA_VISIBLE_DEVICES=${gpuid}  python -u val_cylinder_asym_ood.py \
--config 'config/semantickitti_ood_final.yaml'