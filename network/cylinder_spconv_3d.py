# -*- coding:utf-8 -*-
# author: Xinge
# @file: cylinder_spconv_3d.py

from torch import nn

REGISTERED_MODELS_CLASSES = {}


def register_model(cls, name=None):
    global REGISTERED_MODELS_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_MODELS_CLASSES, f"exist class: {REGISTERED_MODELS_CLASSES}"
    REGISTERED_MODELS_CLASSES[name] = cls
    return cls


def get_model_class(name):
    global REGISTERED_MODELS_CLASSES
    assert name in REGISTERED_MODELS_CLASSES, f"available class: {REGISTERED_MODELS_CLASSES}"
    return REGISTERED_MODELS_CLASSES[name]


@register_model
class cylinder_asym(nn.Module):
    def __init__(self,
                 cylin_model,
                 segmentator_spconv,
                 sparse_shape,
                 ):
        super().__init__()
        self.name = "cylinder_asym"

        self.cylinder_3d_generator = cylin_model

        self.cylinder_3d_spconv_seg = segmentator_spconv

        self.sparse_shape = sparse_shape

    def forward(self, train_pt_fea_ten, train_vox_ten, batch_size):
        coords, features_3d = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten)

        spatial_features = self.cylinder_3d_spconv_seg(features_3d, coords, batch_size)

        return spatial_features

    def forward_dummy(self, train_pt_fea_ten, train_vox_ten, batch_size, ood_num):
        coords, features_3d = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten)

        spatial_features = self.cylinder_3d_spconv_seg.forward_dummy(features_3d, coords, batch_size, ood_num)

        return spatial_features

    def forward_dummy_2(self, train_pt_fea_ten, train_vox_ten, batch_size, ood_num, point_label_tensor):
        coords, features_3d = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten)

        spatial_features = self.cylinder_3d_spconv_seg.forward_dummy_2(features_3d, coords, batch_size, ood_num, point_label_tensor)

        return spatial_features

    def forward_dummy_3(self, train_pt_fea_ten, train_vox_ten, batch_size, ood_num):
        coords, features_3d = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten)

        spatial_features = self.cylinder_3d_spconv_seg.forward_dummy_3(features_3d, coords, batch_size, ood_num)

        return spatial_features

    def forward_dummy_4(self, train_pt_fea_ten, train_vox_ten, batch_size, ood_num):
        coords, features_3d = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten)

        spatial_features = self.cylinder_3d_spconv_seg.forward_dummy_4(features_3d, coords, batch_size, ood_num)

        return spatial_features

    def forward_dummy_final(self, train_pt_fea_ten, train_vox_ten, batch_size, ood_num):
        coords, features_3d = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten)

        spatial_features = self.cylinder_3d_spconv_seg.forward_dummy_final(features_3d, coords, batch_size, ood_num)

        return spatial_features

    def forward_dummy_upper(self, train_pt_fea_ten, train_vox_ten, batch_size, ood_num):
        coords, features_3d = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten)

        spatial_features = self.cylinder_3d_spconv_seg.forward_dummy_upper(features_3d, coords, batch_size, ood_num)

        return spatial_features

    def forward_DML(self, train_pt_fea_ten, train_vox_ten, batch_size):
        coords, features_3d = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten)

        spatial_features = self.cylinder_3d_spconv_seg.forward_DML(features_3d, coords, batch_size)

        return spatial_features

    def forward_dropout(self, train_pt_fea_ten, train_vox_ten, batch_size):
        coords, features_3d = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten)

        spatial_features = self.cylinder_3d_spconv_seg.forward_dropout(features_3d, coords, batch_size)

        return spatial_features

    def forward_dropout_eval(self, train_pt_fea_ten, train_vox_ten, batch_size):
        coords, features_3d = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten)

        spatial_features = self.cylinder_3d_spconv_seg.forward_dropout_eval(features_3d, coords, batch_size)

        return spatial_features

    def forward_incremental(self, train_pt_fea_ten, train_vox_ten, batch_size, incre_cls=None):
        coords, features_3d = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten)

        spatial_features = self.cylinder_3d_spconv_seg.forward_incremental(features_3d, coords, batch_size, incre_cls)

        return spatial_features