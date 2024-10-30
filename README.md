# nn_laser_spot_detection

![logic scheme](./scheme/scheme.png)

Detect (and practically track) in the ROS world a laser spot emitted from a common laser pointer.  


## Requirments:
- It should work on cpu only pc, but gpu (only nvidia) is preferrable
- Pytorch with matching version of cuda. conda (please miniconda) can be used but I probably you have to install also ros in the environent to run all
- ROS (1) 
- If using yolov5, their requirments (see setup and running)


## Setup and running
The tracking comprises two main nodes: 2D_detection for 2D neural network detection and 3D_matching for pixel-point matching and TF broadcasting.  

In brief:  
The 2D_detection node will subscribe to `/$(arg camera)/$(arg image)/$(arg transport)>`  
The 3D_tracking to `/$(arg point_cloud)`
A TF will be broadcasted to the ROS tf tree from the point cloud reference to the `$(arg laser_spot_frame)`.

- Indentify the model/weights you want. Some are provided at [https://zenodo.org/records/10471835](https://zenodo.org/records/10471835). In any case they should have been trained on laser spots, obviously. Put the model you want to use in a folder, `models` folder of this repo is the default.

- [Optional, but suggested] If Yolov5 is used, better to clone their [repo](https://github.com/ultralytics/yolov5/), and provide its path to the `yolo_path` argument. Otherwise, pytorch will download it every time (since the default is "ultralytics/yolov5"). If cloning, go in the folder and install the requirments: `pip install -r requirements.txt`.

- Run the launch file: 
  `roslaunch nn_laser_spot_tracking laser3DTracking.launch model_name:=<> camera:=<> point_cloud:=<>`

### Launch File Arguments
#### Required
- **`model_name`**: Name of the neural network model, a ".pt" file, located in `model_path` folder
- **`camera`**: Camera "ROS" name. This is used to define the topic where images are published. It is the root of the image topic name.
- **`point_cloud`**: The name of the topic where the PointCloud is published (`pcl::PointCloud<pcl::PointXYZ>`)

#### Optional
- **`model_path`** (default: "$(find nn_laser_spot_tracking)/models/"): Path to the neural network model directory.
- **`yolo_path`** (default "ultralytics/yolov5"): Path to the yolo repo, better to download it so you have it locally.
- **`image`** (default: "color/image_raw"): Image topic name.
- **`transport`** (default: "compressed"): Image transport type.
- **`dl_rate`** (default: 30): Rate of the 2D_detection node. Note that inference part is blocking, so the node may not reach this rate
- **`tracking_rate`** (default: 100): Rate of the 3D_matching node.

- **`detection_confidence_threshold`** (default: 0.55): Confidence threshold for 2D detections.
- **`cloud_detection_max_sec_diff`** (default: 4): Maximum timestamp difference between 2D detections and point clouds. If it is bigger, results are discarded.
- **`position_filter_enable`** (default: true): Enable laser spot position filtering to smooth erratic movements of the laser spot.
- **`position_filter_bw`** (default: 9): Bandwidth of the position filter.
- **`position_filter_damping`** (default: 1): Damping factor for the position filter.
These are settable online with a [ddynamic reconfigure](https://github.com/pal-robotics/ddynamic_reconfigure) server.

- **`keypoint_topic`** (default: "/nn_laser_spot_tracking/detection_output_keypoint"): Topic with which 2D_tracking and 3D_matching communicate. Better to not touch.
- **`laser_spot_frame`** (default: "laser_spot_frame"): Frame name for laser spot.
- **`pub_out_images`** (default: true): Publish 2D images with rectangle on the detected laser spot.
- **`pub_out_images_all_keypoints`** (default: false): Publish RGB 2D with all keypoints (not only the best one above the confidence threshold).
- **`pub_out_images_topic`** (default: "/detection_output_img"): Topic name for publishing output images.
- **`gdb`** (default: false): Enable GDB debugging.
- **`rviz`** (default: false): Launch RViz visualization.

## Training new models
- See https://github.com/ADVRHumanoids/nn_laser_spot_tracking/tree/master/training

## Data
- Image dataset: [https://zenodo.org/records/10471811](https://zenodo.org/records/10471811)
- Trained models: [https://zenodo.org/records/10471835](https://zenodo.org/records/10471835)

## Paper
[https://ieeexplore.ieee.org/document/10602529](https://ieeexplore.ieee.org/document/10602529)
```
@ARTICLE{10602529,
  author={Torielli, Davide and Bertoni, Liana and Muratore, Luca and Tsagarakis, Nikos},
  journal={IEEE Robotics and Automation Letters}, 
  title={A Laser-Guided Interaction Interface for Providing Effective Robot Assistance to People With Upper Limbs Impairments}, 
  year={2024},
  volume={9},
  number={9},
  pages={7653-7660},
  keywords={Robots;Lasers;Task analysis;Keyboards;Magnetic heads;Surface emitting lasers;Grippers;Human-robot collaboration;physically assistive devices;visual servoing},
  doi={10.1109/LRA.2024.3430709}}
```
