# Training

Training part is done by adapting tutorials and repos around the web, so some additional files must be 
dowloaded as per instructions

## Dataset
Laser spot annotated image dataset is availalbe at [https://zenodo.org/records/10471811](https://zenodo.org/records/10471811)  
Two formats are given:
- COCO format (for non-yolo models) as:
  - a folder containing images and annotation folders.    
    - in images, all the images (not divided by train, val, test, this is done by the code)
    - in annotations, an instances_default.json file 

- YOLOv5 pytorch format for YOLOV5 model
  - a folder containing data.yaml file which points to two folders in the same location:
    - train
    - valid
    Both have images and labels folders


## Training Non-YOLO models

Taken from https://github.com/pytorch/vision/tree/main

- Download files from https://github.com/pytorch/vision/tree/main/references/detection and put them in this folder, inside a "detection" folder. 
  They will be used by `TrainModel.py`

- Run `TrainModel.py`. Examples about how to run are given in TrainMain.py

## Training YOLOv5 models

Taken from https://github.com/ultralytics/yolov5/

You can either follow the https://github.com/ultralytics/yolov5/ tutorial (Training part)

OR, clone that repo somewhere and use the `TrainYolo.py`. Example about how to use the MainYolo are in `TrainMain.py`
