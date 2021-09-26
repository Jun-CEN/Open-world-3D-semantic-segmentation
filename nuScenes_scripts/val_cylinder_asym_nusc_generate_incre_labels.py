# -*- coding:utf-8 -*-
# author: Xinge
# @file: train_cylinder_asym.py


import os
import time
import argparse
import sys
sys.path.append("..")
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint, load_checkpoint_1b1

import warnings
from shutil import copyfile

warnings.filterwarnings("ignore")


def main(args):
    pytorch_device = torch.device('cuda:0')

    config_path = args.config_path

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    val_batch_size = val_dataloader_config['batch_size']
    train_batch_size = train_dataloader_config['batch_size']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    model_load_path = train_hypers['model_load_path']
    model_save_path = train_hypers['model_save_path']

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    my_model = model_builder.build(model_config)
    if os.path.exists(model_load_path):
        my_model = load_checkpoint(model_load_path, my_model)
        print('Load checkpoint file successfully!')

    my_model.to(pytorch_device)
    optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])

    loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                   num_class=num_class, ignore_label=ignore_label)

    train_dataset_loader, val_dataset_loader = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size)

    # training
    best_val_miou = 0

    # lr_scheduler.step(epoch)
    my_model.eval()
    hist_list = []
    val_loss_list = []
    pbar = tqdm(total=len(val_dataset_loader))
    global_iter = 0
    with torch.no_grad():
        for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea, idx) in enumerate(
                val_dataset_loader):

            val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                              val_pt_fea]
            val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
            val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)

            predict_labels = my_model(val_pt_fea_ten, val_grid_ten, val_batch_size)
            # aux_loss = loss_fun(aux_outputs, point_label_tensor)
            loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels).detach(), val_label_tensor,
                                  ignore=0) + loss_func(predict_labels.detach(), val_label_tensor)
            uncertainty_scores_logits = -torch.max(predict_labels, dim=1)[0]
            uncertainty_scores_logits = uncertainty_scores_logits.cpu().detach().numpy()
            softmax_layer = torch.nn.Softmax(dim=1)

            uncertainty_scores_softmax = 1 - torch.max(softmax_layer(predict_labels), dim=1)[0]
            uncertainty_scores_softmax = uncertainty_scores_softmax.cpu().detach().numpy()
            predict_labels = torch.argmax(predict_labels, dim=1)
            predict_labels = predict_labels.cpu().detach().numpy()
            # val_grid_ten: [batch, points, 3]
            # val_vox_label: [batch, 480, 360, 32]
            # val_pt_fea_ten: [batch, points, 9]
            # val_pt_labs: [batch, points, 1]
            count = 0
            point_predict = predict_labels[count, val_grid[count][:, 0], val_grid[count][:, 1],val_grid[count][:, 2]].astype(np.int32)
            unknown_clss = [1, 5, 8, 9]
            for unknown_cls in unknown_clss:
                point_predict[point_predict==unknown_cls] = 0
            point_uncertainty_logits = uncertainty_scores_logits[count, val_grid[count][:, 0], val_grid[count][:, 1],val_grid[count][:, 2]]
            point_uncertainty_softmax = uncertainty_scores_softmax[count, val_grid[count][:, 0], val_grid[count][:, 1],val_grid[count][:, 2]]
            idx_s = "%06d" % idx[0]
            # point_uncertainty_logits.tofile(
            #         '/harddisk/jcenaa/nuScenes/predictions/scores_logits_base/' + idx_s + '.label')
            # point_uncertainty_softmax.tofile(
            #     '/harddisk/jcenaa/nuScenes/predictions/scores_softmax_upper/' + idx_s + '.label')
            point_predict.tofile(
                '/harddisk/jcenaa/nuScenes/predictions/predictions_base_train/' + idx_s + '.label')

            pred = np.fromfile('/harddisk/jcenaa/nuScenes/predictions/predictions_base_train/' + idx_s + '.label', dtype=np.int32)
            unknown_clss = [1, 5, 8, 9]
            for unknown_cls in unknown_clss:
                if unknown_cls in np.unique(pred):
                    print(np.unique(pred, return_counts=True))
                    print(pred_file)

            for count, i_val_grid in enumerate(val_grid):
                hist_list.append(fast_hist_crop(predict_labels[
                                                    count, val_grid[count][:, 0], val_grid[count][:, 1],
                                                    val_grid[count][:, 2]], val_pt_labs[count],
                                                unique_label))
            val_loss_list.append(loss.detach().cpu().numpy())
            pbar.update(1)
    iou = per_class_iu(sum(hist_list))
    print('Validation per class iou: ')
    for class_name, class_iou in zip(unique_label_str, iou):
        print('%s : %.2f%%' % (class_name, class_iou * 100))
    val_miou = np.nanmean(iou) * 100
    del val_vox_label, val_grid, val_pt_fea, val_grid_ten

    print('Current val miou is %.3f while the best val miou is %.3f' %
          (val_miou, best_val_miou))
    print('Current val loss is %.3f' %
          (np.mean(val_loss_list)))


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='../config/nuScenes_ood_generate_incre_labels.yaml')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
