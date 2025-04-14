#!/usr/bin/env python3
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import sys

root_dir = "/home/tori/YOLO/"
dataset_name = "laser_home"
model_name = "yolov5l6_e200_b8_tvt302010_laser_v5"

annotation = root_dir + dataset_name + "/annotations/instances_default.json"
img_dir = root_dir + dataset_name + "/images/"
coco = COCO(annotation)    


testing_results = coco.loadRes("results/" + dataset_name+"_"+model_name+"_detections.json")

# Initialize COCOeval
coco_eval = COCOeval(coco, testing_results, iouType='bbox')
coco_eval.params.useCats = 1
#coco_eval.params.iouThrs = [0.001]
#coco_eval.params.imgIds = range(5,7)
coco_eval.params.iouThrs = [0.5]
coco_eval.params.recThrs = [0.5]

# Run evaluation
coco_eval.evaluate()
coco_eval.accumulate()

with open("results/" + dataset_name+"_"+model_name+"_benchmark_result.txt", 'w') as f:
    #redirtect print of the coco_eval_summarize() to a file
    original_stdout = sys.stdout
    sys.stdout = f
    coco_eval.summarize()
    sys.stdout = original_stdout  # Restore to terminal
f.close()