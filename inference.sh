#!/usr/bin/env bash

# Classification

conda activate WCE_classification

python classification/tools/test1.py

python classification/tools/test2.py

conda deactivate

# Instance segmentation

conda activate WCE_instance_seg

python instance_segmentation/test1.py \
    --format-only \
    --options \
    "jsonfile_prefix=./instance_segmentation/work_dirs/test1_internimage_xl/mask_and_bbox_results/instance_segmentation_test1"

python instance_segmentation/test2.py \
    --format-only \
    --options \
    "jsonfile_prefix=./instance_segmentation/work_dirs/test2_internimage_xl/mask_and_bbox_results/instance_segmentation_test2"

conda deactivate