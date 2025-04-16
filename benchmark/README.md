# Benchmarks
In this folder scripts (and utils) are present to test various methods to detect/track a laser spot in a 2D image. Code does not rely on ROS, but on *OpenCV* / *Pytorch* / *pycocotools* (for evaluation).

## Methods
1. [`laser_detect_filter_blob.py`](./laser_detect_filter_blob.py) A classic OpenCV method taken partially from https://github.com/bradmontgomery/python-laser-tracker
  - Filter the image based on HSV value. Considering the task of detecting a *red* laser dot, high values of *hue* are selected, and low values of *saturation* are discarded (*tunable in the code*)
  - On the filtered image, perform a blob detection with the OpenCV [SimpleBlobDetector](https://docs.opencv.org/4.10.0/d0/d7a/classcv_1_1SimpleBlobDetector.html), setting some parameters like filtering by color, area, circularity (*tunable in the code*)
  - All detected blobs are returned, without a specific score

2. [`laser_detect_nn.py`](./laser_detect_nn.py) The main neural network-based [method](../scripts/inferNodeROS.py) used in this repo for the 2D detection, stripped of ROS communications. Check inside for parameters, such as to indicate the NN weights.

## How to evaluate
After running one of the methods above (check the scripts for setting correctly the parameters), run the [`laser_detect_evaluate.py`](./laser_detect_evaluate.py) script (again check for parameters). A `.txt` will be produced with summary of results from the [cocoeval](https://cocodataset.org/#detection-eval) lib.

## Util
[`color_thresholder.py`](./color_thresholder.py) a simple GUI to filter out an image on its HSV color space interactively

## Data
- Image dataset: [https://zenodo.org/records/15230870](https://zenodo.org/records/15230870)
- Trained models: [https://zenodo.org/records/10471835](https://zenodo.org/records/10471835)