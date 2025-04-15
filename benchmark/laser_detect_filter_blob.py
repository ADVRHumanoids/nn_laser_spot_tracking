#!/usr/bin/env python
### FROM https://github.com/bradmontgomery/python-laser-tracker/blob/master/laser_tracker/laser_tracker.py

import cv2
import numpy
import sys
import os
import time
import json
from pycocotools.coco import COCO

class LaserTracker(object):

    def __init__(self, cam_width=640, cam_height=480, hue_min=140, hue_max=179,
                 sat_min=70, sat_max=255, val_min=0, val_max=255, display=True,
                 display_thresholds=False):
        """
        * ``cam_width`` x ``cam_height`` -- This should be the size of the
        image coming from the camera. Default is 640x480.
        HSV color space Threshold values for a RED laser pointer are determined
        by:
        * ``hue_min``, ``hue_max`` -- Min/Max allowed Hue values
        * ``sat_min``, ``sat_max`` -- Min/Max allowed Saturation values
        * ``val_min``, ``val_max`` -- Min/Max allowed pixel values
        If the dot from the laser pointer doesn't fall within these values, it
        will be ignored.
        * ``display_thresholds`` -- if True, additional windows will display
          values for threshold image channels.
        """

        self.cam_width = cam_width
        self.cam_height = cam_height
        self.hue_min = hue_min
        self.hue_max = hue_max
        self.sat_min = sat_min
        self.sat_max = sat_max
        self.val_min = val_min
        self.val_max = val_max
        self.display = display
        self.display_thresholds = display_thresholds

        self.hsv_img = None
        self.hsv_mask = None
        self.thresholded_img = None
        self.keypoints = None

        self.lower_thres = numpy.array([self.hue_min, self.sat_min, self.val_min])
        self.upper_thres = numpy.array([self.hue_max, self.sat_max, self.val_max])

        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 255
        params.filterByArea = False
        params.minArea = 1
        params.maxArea = 50
        params.filterByCircularity = True
        params.minCircularity = 0.5
        params.filterByInertia = False
        params.filterByConvexity = False

        self.blob_detector = cv2.SimpleBlobDetector_create(params)
        
        if self.display:
            self.setup_windows()

    def detect(self, frame):
        
        tic_py = time.time()

        self.hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        ### Threshold HSV colors
        self.hsv_mask = cv2.inRange(self.hsv_img, self.lower_thres, self.upper_thres)
        self.thresholded_img = cv2.bitwise_and(frame, frame, mask=self.hsv_mask)

        ### Blob detection
        self.keypoints = self.blob_detector.detect(self.thresholded_img)

        toc_py = time.time()
        detect_time = toc_py - tic_py

        if self.display:

            im_with_keypoints = cv2.drawKeypoints(frame, self.keypoints, numpy.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            for point in self.keypoints:
                # Draw the keypoint on the image
                print(f"keypoint res: {point.response}")  
                print(f"keypoint size : {point.size}")  
                print(f"keypoint angle: {point.angle}")  
                print(f'keypoint point: {point.pt[0]}; {point.pt[1]}')
                cv2.rectangle(im_with_keypoints, 
                       (round(point.pt[0]-point.size/2), round(point.pt[1]-point.size/2)), 
                       (round(point.pt[0]+point.size/2), round(point.pt[1]+point.size/2)), 
                       (255, 0, 0), 
                       1)
                
            print(f'time: ' + str(detect_time))
            cv2.imshow('im_with_keypoints', im_with_keypoints)
            thresholded_img_with_keypoints = cv2.drawKeypoints(self.thresholded_img, self.keypoints, numpy.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow('thresholded_img_with_keypoints', thresholded_img_with_keypoints)

            if self.display_thresholds:
                # split the video frame into color channels
                h, s, v = cv2.split(self.thresholded_img)
                cv2.imshow('hsv_image', self.hsv_img)
                cv2.imshow('Hue', h)
                cv2.imshow('Saturation', s)
                cv2.imshow('Value', v)

            cv2.waitKey()

        return self.keypoints, detect_time

    def create_and_position_window(self, name, xpos, ypos):
        """Creates a named widow placing it on the screen at (xpos, ypos)."""
        # Create a window
        cv2.namedWindow(name)
        # Resize it to the size of the camera image
        cv2.resizeWindow(name, self.cam_width, self.cam_height)
        # Move to (xpos,ypos) on the screen
        cv2.moveWindow(name, xpos, ypos)
        
    def setup_windows(self):
        sys.stdout.write("Using OpenCV version: {0}\n".format(cv2.__version__))

        # create output windows
        self.create_and_position_window('im_with_keypoints',
                                        10 + self.cam_width, 0)        
        self.create_and_position_window('thresholded_img_with_keypoints',
                                        20 + self.cam_width, 0)
                                        
        if self.display_thresholds:
            self.create_and_position_window('hsv_image', 10, 10)
            self.create_and_position_window('Hue', 20, 20)
            self.create_and_position_window('Saturation', 30, 30)
            self.create_and_position_window('Value', 40, 40)

if __name__ == '__main__':

    root_dir = "/home/tori/YOLO/"
    dataset_name = "laser_home"
    model_name = "filter_blob"

    annotation = root_dir + dataset_name + "/annotations/instances_default.json"
    img_dir = root_dir + dataset_name + "/images/"
    coco = COCO(annotation)
    tracker = LaserTracker(cam_width=1280, cam_height=720, display=False,
                           display_thresholds=False)
    
    detections = []
    det_times = []
    mean_det_time = 0.0
    for img_id in coco.getImgIds():
        filename = coco.loadImgs(img_id)[0]['file_name']
        img = cv2.imread(os.path.join(img_dir,filename))
        points, det_time = tracker.detect(img)

        for p in points:
            detections.append({
                "image_id": img_id,
                "category_id": 1,
                "bbox": [round((p.pt[0]-p.size/2.0),2), round((p.pt[1]-p.size/2.0),2), round(p.size,2), round(p.size,2)],
                "score": 1.0,
            })

        det_times.append({
            "image_id": img_id,
            "detect_time": det_time
        })
        mean_det_time += det_time

    cv2.destroyAllWindows()

    print(f"Mean time: {round(mean_det_time/len(coco.getImgIds()),6)}")

    with open("results/" + dataset_name+"_"+model_name+"_detections.json", 'w') as f:
        json.dump(detections, f)   

    with open("results/" + dataset_name+"_"+model_name+"_detections_time.json", 'w') as f:
        json.dump(det_times, f)     

