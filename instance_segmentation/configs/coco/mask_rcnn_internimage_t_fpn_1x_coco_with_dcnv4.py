# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance_WCE.py',
    '../_base_/schedules/schedule_3x.py',
    '../_base_/default_runtime.py'
]
pretrained = 'https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_t_1k_224.pth'
model = dict(
    backbone=dict(
        _delete_=True,
        type='InternImage',
        core_op='DCNv3',
        channels=64,
        depths=[4, 4, 18, 4],
        groups=[4, 8, 16, 32],
        mlp_ratio=4.,
        drop_path_rate=0.2,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=1.0,
        post_norm=False,
        with_cp=False,
        out_indices=(0, 1, 2, 3),
        use_dcn_v4_op=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=5),
    roi_head=dict(
        bbox_head=dict(
            num_classes=1,
        ),
        mask_head=dict(
            num_classes=1,
        )
    ),
    train_cfg=dict(
        rpn_proposal=dict(
            nms=dict(type='nms', iou_threshold=0.4),
        )
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.01),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.2),
            max_per_img=100,
            mask_thr_binary=0.2))
)

optimizer = dict(
    _delete_=True, type='AdamW', lr=0.0001, weight_decay=0.05,
    constructor='CustomLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=30, layer_decay_rate=1.0,
                       depths=[4, 4, 18, 4]))
optimizer_config = dict(grad_clip=None)

evaluation = dict(interval=1, save_best='auto', metric=['bbox', 'segm'])

checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=5,
    save_last=True,
)

runner = dict(type='EpochBasedRunner', max_epochs=1000)

log_config = dict(
    interval=50, 
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False)
    ])  

