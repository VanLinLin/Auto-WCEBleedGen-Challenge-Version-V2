#!/usr/bin/env bash

# Classification
conda activate WCE_classification

python classification/tools/predict.py

conda deactivate

# Instance segmentation and CAM
conda activate WCE_instance_seg

python instance_segmentation/CAM_neck_avg_channel.py \
    --save_path inference_pipeline \
    --img_path inference_pipeline/bleeding \
    --combine

conda deactivate