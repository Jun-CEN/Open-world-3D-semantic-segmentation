
# Open-world semantic segmentation for Lidar points

## Installation

### Requirements
- PyTorch >= 1.2 
- yaml
- Cython
- [torch-scatter](https://github.com/rusty1s/pytorch_scatter)
- [nuScenes-devkit](https://github.com/nutonomy/nuscenes-devkit) (optional for nuScenes)
- [spconv](https://github.com/traveller59/spconv) (tested with spconv==1.2.1 and cuda==10.2)

## Data Preparation

### SemanticKITTI
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
    ├──sequences
        ├── 00/           
        │   ├── velodyne/	
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 
        |       ├── 000000.label
        |       ├── 000001.label
        |       └── ...
        ├── 08/ # for validation
        ├── 11/ # 11-21 for testing
        └── 21/
	    └── ...
```

### nuScenes
```
./
├── 
├── ...
└── lidarseg/
    ├──v1.0-trainval/
    ├──v1.0-mini/
    ├──v1.0-test/
    ├──nuscenes_infos_train.pkl
    ├──nuscenes_infos_val.pkl
    ├──nuscenes_infos_test.pkl
└── lidarseg/
    ├──v1.0-trainval/
    ├──v1.0-mini/
    ├──v1.0-test/
```

## Training for SemanticKITTI
### Naive method
```
./train_naive.sh
```
### Upper bound
```
./train_upper.sh
```
### Classifier placeholder
- Change the path of pretrained naive model in `/config/semantickitti_ood_basic.yaml`, line 63.

- Change the coefficient lamda_1 in `/config/semantickitti_ood_basic.yaml`, line 70.

- Change the dummy classifier number in `/train_cylinder_asym_ood_basic.py`, line 198.
```
./train_ood_basic.sh
```
### Data placeholder

- Change the path of pretrained naive model in `/config/semantickitti_ood_final.yaml`, line 63.

- Change lamda_1, lamda_2 in `/config/semantickitti_ood_final.yaml`, line 70, 71.

- Change the dummy classifier number in `/train_cylinder_asym_ood_final.py`, line 198.
```
./train_ood_final.sh
```
## Evaluation for SemanticKITTI
We save the in-distribution prediction labels and uncertainty scores for every points in the validation set, 
and these files will be used to calculate the closed-set mIoU and open-set metrics including AUPR, AURPC, and FPR95.
### MSP/Maxlogit
- Change the trained model path (Naive method) in `/config/semantickitti.yaml`, line 63.

- Change the saving path of in-distribution prediction results and uncertainty scores in `val_cylinder_asym.py`, line 112, 114, 116.
```
./val.sh
```

### Upper bound
- Change the trained model path (Placeholder method) in `/config/semantickitti.yaml`, line 63.

- Change the saving path of in-distribution prediction results and uncertainty scores in `val_cylinder_asym_upper.py`, line 115, 117.

```
./val_upper.sh
```

### Classifier/Data placeholder
- Change the trained model path (Placeholder method) in `/config/semantickitti_ood_final.yaml`, line 63.

- Change the saving path of in-distribution prediction results and uncertainty scores in `val_cylinder_asym_ood.py`, line 124, 125.

```
./val_ood.sh
```
## Training for nuScenes
### Naive method
```
./train_nusc_naive.sh
```
### Upper bound
```
./train_nusc.sh
```
### Classifier placeholder
- Change the path of pretrained naive model in `/config/nuScenes_ood_basic.yaml`, line 63.

- Change the coefficient lamda_1 in `/config/nuScenes_ood_basic.yaml`, line 70.

- Change the dummy classifier number in `/train_cylinder_asym_nuscenes_ood_basic.py`, line 197.
```
./train_nusc_ood_basic.sh
```
### Data placeholder

- Change the path of pretrained naive model in `/config/nuScenes_ood_final.yaml`, line 63.

- Change lamda_1, lamda_2 in `/config/nuScenes_ood_final.yaml`, line 70, 71.

- Change the dummy classifier number in `/train_cylinder_asym_nuscenes_ood_final.py`, line 197.
```
./train_nusc_ood_final.sh
```
## Evaluation for nuScenes

### MSP/Maxlogit
- Change the trained model path (Naive method) in `/config/nuScenes.yaml`, line 63.

- Change the saving path of in-distribution prediction results and uncertainty scores in `val_cylinder_asym_nusc.py`, line 112, 114, 116.
```
./val_nusc.sh
```

### Upper bound
- Change the trained model path (Naive method) in `/config/nuScenes.yaml`, line 63.

- Change the saving path of in-distribution prediction results and uncertainty scores in `val_cylinder_asym_nusc_upper.py`, line 121, 123.
```
./val_nusc_upper.sh
```

### Classifier/Data placeholder
- Change the trained model path (Placeholder method) in `/config/nuScenes_ood_final.yaml`, line 63.

- Change the saving path of in-distribution prediction results and uncertainty scores in `val_cylinder_asym_nusc_ood.py`, line 125, 126.

```
./val_nusc_ood.sh
```