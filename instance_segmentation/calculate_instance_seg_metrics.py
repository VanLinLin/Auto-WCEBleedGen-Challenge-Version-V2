##The code of eval_coco.py
import pycocotools.coco as coco
from utils.metrics_calculator import COCOeval

# Prediction result
results = 'instance_segmentation/ensemble_results/test1_affirmative.bbox.json'
task = results.split('.')[1]  # get the task string e.g. bbox or segm

# Ground truth
anno = 'instance_segmentation/data/WCEBleedGen_v2/instance_seg_img_test1/coco_annotation/anno_test1.json'

coco_anno = coco.COCO(anno)
coco_dets = coco_anno.loadRes(results)
coco_eval = COCOeval(coco_anno, coco_dets, f"{task}")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()