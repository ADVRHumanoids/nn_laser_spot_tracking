# Benchmarks
In this folder scripts (and utils) are present to test various methods to detect/track a laser spot in a 2D image. Code not rely on ROS, but on OpenCV / Pytorch only

1. `laser_detect_filter_blob.py`. A classic OpenCV method taken partially from https://github.com/bradmontgomery/python-laser-tracker
  - Filter the image based on HSV value. Considering the task of detecting a *red* laser dot, high values of *hue* are selected, and low values of *saturation* are discarded
  - On the filtered image, perform a blob detection with the OpenCV [SimpleBlobDetector](https://docs.opencv.org/4.10.0/d0/d7a/classcv_1_1SimpleBlobDetector.html), setting some parameters like filtering by color, area, circularity.
  - All detected blobs are returned, without a specific score

2. `laser_detect_nn.py` *TODO*

## How to run
*TODO*