#!/usr/bin/env bash

# Classification

conda activate WCE_classification

python classification/tools/test1.py

python classification/tools/test2.py

conda deactivate

# Instance segmentation

conda activate WCE_instance_seg

python instance_segmentation/test1.py

python instance_segmentation/test2.py

conda deactivate