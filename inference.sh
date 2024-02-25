#!/usr/bin/env bash

echo Start inference, task: classification, dataset: test1

conda activate WCE_classification

echo Activate classification environment!

python classification/tools/test1.py

python classification/tools/test2.py