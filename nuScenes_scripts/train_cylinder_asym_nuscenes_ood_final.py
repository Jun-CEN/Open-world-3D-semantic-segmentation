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
import spconv

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_nuScenes_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint, load_checkpoint_1b1
from torch.utils.tensorboard import SummaryWriter

import warnings

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
    model_latest_path = train_hypers['model_latest_path']

    lamda_1 = train_hypers['lamda_1']
    lamda_2 = train_hypers['lamda_2']

    SemKITTI_label_name = get_nuScenes_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    my_model = model_builder.build(model_config)

    my_model.cylinder_3d_spconv_seg.logits2 = spconv.SubMConv3d(4 * 32, args.dummynumber, indice_key="logit",
                                                                kernel_size=3, stride=1, padding=1,
                                                                bias=True).to(pytorch_device)

    if os.path.exists(model_load_path):
        my_model = load_checkpoint_1b1(model_load_path, my_model)

    my_model.to(pytorch_device)
    optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])

    loss_func_train, lovasz_softmax = loss_builder.build_ood(wce=True, lovasz=True,
                                                             num_class=num_class, ignore_label=ignore_label, weight=lamda_2)
    loss_func_val, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                       num_class=num_class, ignore_label=ignore_label)

    train_dataset_loader, val_dataset_loader = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size)

    # training
    epoch = 0
    best_val_miou = 0
    my_model.train()
    global_iter = 0
    check_iter = train_hypers['eval_every_n_steps']
    writer = SummaryWriter('/harddisk/jcenaa/nuScenes/log')
    while epoch < train_hypers['max_num_epochs']:
        loss_list = []
        pbar = tqdm(total=len(train_dataset_loader))
        time.sleep(10)
        for i_iter, (_, train_vox_label, train_grid, _, train_pt_fea) in enumerate(train_dataset_loader):
            if global_iter % check_iter == 0 and epoch >= 0:
                pbar_val = tqdm(total=len(val_dataset_loader))
                my_model.eval()
                hist_list = []
                val_loss_list = []
                with torch.no_grad():
                    for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea, idx) in enumerate(
                            val_dataset_loader):

                        val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                          val_pt_fea]
                        val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
                        val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)

                        predict_labels = my_model(val_pt_fea_ten, val_grid_ten, val_batch_size)
                        loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels).detach(), val_label_tensor,
                                              ignore=0) + loss_func_val(predict_labels.detach(), val_label_tensor)
                        predict_labels = torch.argmax(predict_labels, dim=1)
                        predict_labels = predict_labels.cpu().detach().numpy()
                        count=0
                        print(np.unique(predict_labels[
                                                                count, val_grid[count][:, 0], val_grid[count][:, 1],
                                                                val_grid[count][:, 2]], return_counts=True))
                        print(np.unique(val_pt_labs[count], return_counts=True))
                        for count, i_val_grid in enumerate(val_grid):
                            hist_list.append(fast_hist_crop(predict_labels[
                                                                count, val_grid[count][:, 0], val_grid[count][:, 1],
                                                                val_grid[count][:, 2]], val_pt_labs[count],
                                                            unique_label))
                        val_loss_list.append(loss.detach().cpu().numpy())
                        pbar_val.update(1)
                my_model.train()
                iou = per_class_iu(sum(hist_list))
                print('Validation per class iou: ')
                for class_name, class_iou in zip(unique_label_str, iou):
                    print('%s : %.2f%%' % (class_name, class_iou * 100))
                val_miou = np.nanmean(iou) * 100
                del val_vox_label, val_grid, val_pt_fea, val_grid_ten

                torch.save(my_model.state_dict(), model_latest_path)
                # save model if performance is improved
                if best_val_miou < val_miou:
                    best_val_miou = val_miou
                    torch.save(my_model.state_dict(), model_save_path)

                print('Current val miou is %.3f while the best val miou is %.3f' %
                      (val_miou, best_val_miou))
                print('Current val loss is %.3f' %
                      (np.mean(val_loss_list)))

            train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in train_pt_fea]
            train_vox_ten = [torch.from_numpy(i).to(pytorch_device) for i in train_grid]
            point_label_tensor = train_vox_label.type(torch.LongTensor).to(pytorch_device)
            unknown_clss = [1,5,8,9]
            for unknown_cls in unknown_clss:
                point_label_tensor[point_label_tensor == unknown_cls] = 0

            # forward + backward + optimize
            coor_ori, output_normal_dummy = my_model.forward_dummy_final(train_pt_fea_ten, train_vox_ten,
                                                                         train_batch_size,
                                                                         args.dummynumber)
            voxel_label_origin = point_label_tensor[coor_ori.permute(1, 0).chunk(chunks=4, dim=0)]

            loss_normal = loss_func_train(output_normal_dummy, point_label_tensor)

            output_normal_dummy = output_normal_dummy.permute(0, 2, 3, 4, 1)
            output_normal_dummy = output_normal_dummy[coor_ori.permute(1, 0).chunk(chunks=4, dim=0)].squeeze()
            index_tmp = torch.arange(0, coor_ori.shape[0]).unsqueeze(0).cuda()
            voxel_label_origin[voxel_label_origin == 17] = 0
            index_tmp = torch.cat([index_tmp, voxel_label_origin], dim=0)
            output_normal_dummy[index_tmp.chunk(chunks=2, dim=0)] = -1e9
            label_dummy = torch.ones(output_normal_dummy.shape[0]).type(torch.LongTensor).cuda() * 17
            label_dummy[voxel_label_origin.squeeze() == 0] = 0
            loss_dummy = loss_func_train(output_normal_dummy, label_dummy)

            writer.add_scalar('Loss/loss_normal', loss_normal.item(), global_iter)
            writer.add_scalar('Loss/loss_dummy', loss_dummy.item(), global_iter)
            # print(point_label_tensor.shape) # [bt, 480, 360, 32]
            # print(outputs.shape) # [bt, 20, 480, 360, 32]
            loss = loss_normal + lamda_1 * loss_dummy
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

            if global_iter % 1000 == 0:
                if len(loss_list) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter, np.mean(loss_list)))
                else:
                    print('loss error')

            optimizer.zero_grad()
            pbar.update(1)
            global_iter += 1
            if global_iter % check_iter == 0:
                if len(loss_list) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter, np.mean(loss_list)))
                else:
                    print('loss error')
        pbar.close()
        epoch += 1


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='../config/nuScenes_ood_final.yaml')
    parser.add_argument('--dummynumber', default=3, type=int, help='number of dummy label.')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
