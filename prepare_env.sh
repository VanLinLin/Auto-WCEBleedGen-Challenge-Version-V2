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

conda env create -f instance_segmentation/WCE_instance_segmentation.yaml

conda activate WCE_instance_seg

conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia

pip install -U openmim

mim install mmcv-full==1.5.0

pip install timm==0.6.11 mmdet==2.28.1

pip install opencv-python termcolor yacs pyyaml scipy

conda install -c conda-forge cudatoolkit-dev -y

cd instance_segmentation/ops_dcnv3 || exit

bash make.sh

cd ../..

conda deactivate