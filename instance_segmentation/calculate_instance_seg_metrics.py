##The code of eval_coco.py
import pycocotools.coco as coco
import argparse
from utils.metrics_calculator import COCOeval


def parse_args():
    parser = argparse.ArgumentParser("The instance segmentation metrics calculaotr")

    parser.add_argument('--result_json',
                        default='instance_segmentation/ensemble_results/test1_affirmative.bbox.json',
                        help='result bbox or segm json file')

    parser.add_argument('--GT_json',
                        default='instance_segmentation/data/WCEBleedGen_v2/instance_seg_img_test1/coco_annotation/anno_test1.json',
                        help='the ground truth of dataset')
    return parser.parse_args()
    
def main():
    args = parse_args()

    task = args.result_json.split('.')[1]  # get the task string e.g. bbox or segm

    coco_anno = coco.COCO(args.GT_json)
    coco_dets = coco_anno.loadRes(args.result_json)
    coco_eval = COCOeval(coco_anno, coco_dets, f"{task}")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
if __name__ == '__main__':
    main()