# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
_base_ = [
    './mask_rcnn_internimage_t_fpn_1x_coco_with_dcnv4.py'
]

pretrained = 'https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_xl_22k_192to384.pth'
model = dict(
    backbone=dict(
        _delete_=True,
        type='InternImage',
        core_op='DCNv3',
        channels=192,
        depths=[5, 5, 24, 5],
        groups=[12, 24, 48, 96],
        mlp_ratio=4.,
        drop_path_rate=0.6,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=2.0,
        post_norm=True,
        with_cp=False,
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        in_channels=[192, 384, 768, 1536]),
)

# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/WCEBleedGen_v2/bleeding/'
classes = ('bleeding',)
data = dict(samples_per_gpu=20)

optimizer = dict(
    _delete_=True, type='AdamW', lr=0.0001, weight_decay=0.05,
    constructor='CustomLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=39, layer_decay_rate=0.94,
                       depths=[5, 5, 24, 5]))

work_dir = './instance_segmentation/work_dirs/test1_internimage_xl'