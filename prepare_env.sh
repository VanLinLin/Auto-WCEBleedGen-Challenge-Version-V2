#!/usr/bin/env bash

# Classification

echo Start create classification environment!

conda env create -f classification/WCE_classification_env.yaml

conda activate WCE_classification

pip install -U openmim

mim install "mmpretrain>=1.0.0rc8"

mim install "mmengine==0.10.3"

conda deactivate

echo Finish creating classification environment!

# Instance segmentation

echo Start create instance segmentation environment!

conda env create -f classification/WCE_instance_seg_env.yaml

conda activate WCE_instance_seg_env

conda install -c conda-forge cudatoolkit-dev

conda activate WCE_instance_seg_env

pip install -U openmim

mim install mmcv-full==1.5.0

pip install timm==0.6.11 mmdet==2.28.1

bash instance_segmentation/ops_dcnv3/make.sh

conda deactivate