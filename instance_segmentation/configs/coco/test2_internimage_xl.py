# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
_base_ = [
    './test1_internimage_xl.py'
]


data_root = 'instance_segmentation/data/WCEBleedGen_v2/'

data = dict(
    test=dict(
        ann_file=data_root + 'instance_seg_img_test2/coco_annotation/anno_test2.json',
        img_prefix=data_root +'instance_seg_img_test2/Images/',))

work_dir = './instance_segmentation/work_dirs/test2_internimage_xl'