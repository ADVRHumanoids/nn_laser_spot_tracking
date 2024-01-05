#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 10:48:52 2022

@author: tori
with ideas taken from https://github.com/vvasilo/yolov3_pytorch_ros/blob/master/src/yolov3_pytorch_ros/detector.py
"""
#import sys
import os
import numpy as np
import time


# Pytorch stuff
import torch
import torchvision.transforms
#from PIL import Image as PILImage

#Opencv stuff
import cv2
from cv_bridge import CvBridge, CvBridgeError

# ROS imports
import rospy
from sensor_msgs.msg import CompressedImage as ROSCompressedImage
from sensor_msgs.msg import Image as ROSImage
from sensor_msgs.msg import CameraInfo

from nn_laser_spot_tracking.msg import KeypointImage

class getCameraInfo:
    
    cam_info = {}
    
    def __init__(self, image_info_topic):
        self.sub = rospy.Subscriber(image_info_topic, CameraInfo, self.__callback)
        rospy.loginfo("waiting for camerainfo...")
        rospy.wait_for_message(image_info_topic, CameraInfo, timeout=None)
        rospy.loginfo("... camerainfo arrived")

    def __callback(self, msg):
        self.cam_info["width"] = msg.width
        self.cam_info["height"] = msg.height
        self.sub.unregister()

class GenericModel : 
    model = None
    device = None
    _transform_chain = None
    tensor_images = []
    
    def __init__(self):
        pass
        
    def initialize(self, model_path_name, yolo_path="", device='gpu'):
        pass
        
    def infer(self, cv_image_input):
        pass
    
class NoYoloModel(GenericModel) : 
    
    def __init__(self):
        super().__init__()
        
    def __process_image(self, cv_image_input):
        
        #pil_image_input = PILImage.fromarray(self.cv_image_input) #img as opencv
        #pil_image_input.show()
        
        self.tensor_images = [(self._transform_chain(cv_image_input).to(self.device))]        

        #beh = torchvision.transforms.functional.to_pil_image(self.tensor_images[0], "RGB")
       # beh.show()    
        
    def initialize(self, model_path_name, yolo_path="", device='gpu'):
        
        if device == 'cpu' :
            self.device = torch.device('cpu')
            self.model = torch.load(model_path_name, map_location=torch.device('cpu'))
     
        elif device == 'gpu' :
            self.device = torch.device('cuda')
            self.model = torch.load(model_path_name)
       
        else:
            raise Exception("Invalid device")   
            
        # wants a tensor
        self._transform_chain = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float),
            ])
        
        self.model.eval()
        self.model.to(self.device)
        
    def infer(self, cv_image_input):
        
        self.__process_image(cv_image_input)
        out = self.model(self.tensor_images)[0]
        
        return out
    
    
class YoloModel(GenericModel) : 
    
    def __init__(self):
        super().__init__()
        
    def initialize(self, model_path, yolo_path="", device='gpu'):

        if device == 'cpu' :
            self.device = torch.device('cpu')
            self.model = torch.hub.load(yolo_path, 'custom', source='local', path=model_path, force_reload=True, map_location=torch.device('cpu'))

        elif device == 'gpu' :
            self.device = torch.device('cuda')
            self.model = torch.hub.load(yolo_path, 'custom', source='local', path=model_path, force_reload=True)
       
        else:
            raise Exception("Invalid device " + device)   
            
        # wants a tensor?
        self._transform_chain = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float),
            ])
        
        self.model.eval()
        self.model.to(self.device)
        
    def infer(self, cv_image_input):
        
        self.tensor_images = [(self._transform_chain(cv_image_input).to(self.device))]        
        
        out_yolo = self.model(cv_image_input)
        
        #out_yolo.print()
        #print(out_yolo.xyxy)
        
        self.out = {
            'boxes': torch.tensor(torch.zeros(len(out_yolo.xyxy[0]), 4), device=self.device),
            'labels': torch.tensor(torch.zeros(len(out_yolo.xyxy[0])), dtype=torch.int32, device=self.device),
            'scores': torch.tensor(torch.zeros(len(out_yolo.xyxy[0])), device=self.device)
        }

        # xyxy has is array with elements of format : 
        # xmin    ymin    xmax   ymax  confidence  class
        for i in range(0, len(out_yolo.xyxy[0])) :
            self.out['boxes'][i] = out_yolo.xyxy[0][i][0:4]
            self.out['scores'][i] = out_yolo.xyxy[0][i][4]
            self.out['labels'][i] = out_yolo.xyxy[0][i][5].int()
        
        #print(self.out)

        return self.out

class DetectorManager():
    
    ros_image_input = ros_image_output = None
    cv_image_input = cv_image_output = np.zeros((100,100,3), np.uint8)
    new_image = False
    model_helper = None
    out = {'scores' : []}
    best_index = -1
    inference_stamp = None
    
    def __init__(self):
        
        self.inference_stamp = rospy.Time.now()

        ### Input Params
        model_path = rospy.get_param('~model_path')
        model_name = rospy.get_param('~model_name')
        yolo_path = rospy.get_param('~yolo_path', "ultralytics/yolov5")

        camera_image_topic = rospy.get_param('~camera_image_topic')
        self.camera_image_transport = rospy.get_param('~transport', 'compressed')
        ros_image_input_topic = camera_image_topic + '/' + self.camera_image_transport

        ## Detection Params
        self.detection_confidence_threshold = rospy.get_param('~detection_confidence_threshold', 0.55)
        
        ### Output Params
        pub_out_keypoint_topic = rospy.get_param('~pub_out_keypoint_topic', "/detection_output_keypoint")
        self.pub_out_images = rospy.get_param('~pub_out_images', True)
        self.pub_out_all_keypoints = rospy.get_param('~pub_out_images_all_keypoints', False)
        pub_out_images_topic = rospy.get_param('~pub_out_images_topic', "/detection_output_img")
        
        #camera_info_topic = rospy.get_param('~camera_info_topic', '/D435_head_camera/color/camera_info')
        #getCameraInfo(camera_info_topic)
        #self.cam_info = getCameraInfo.cam_info
        
        if (model_name.startswith('yolo')) :
            self.model_helper = YoloModel()
        
        else:
            self.model_helper = NoYoloModel()
        
        ############ PYTHORCH STUFF
        model_path_name = os.path.join(model_path, model_name)
        
        rospy.loginfo(f"Using model {model_path_name}")
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            rospy.loginfo("CUDA available, use GPU")
            self.model_helper.initialize(model_path_name, yolo_path, 'gpu')

        else:
            self.device = torch.device('cpu')
            rospy.loginfo("CUDA not available, use CPU") 
            self.model_helper.initialize(model_path_name, yolo_path, 'cpu')
        
        ############ ROS STUFF
        
        self.bridge = CvBridge()
        if self.camera_image_transport == "compressed":
            self.image_sub = rospy.Subscriber(ros_image_input_topic, ROSCompressedImage,
                                              self.__image_clbk, queue_size = 1)
            rospy.loginfo("waiting for camera message...")
            rospy.wait_for_message(ros_image_input_topic, ROSCompressedImage, timeout=None)
            rospy.loginfo("... camera message arrived")
            self.ros_image_input = ROSCompressedImage
            
        else:
            self.image_sub = rospy.Subscriber(ros_image_input_topic, ROSImage,
                                              self.__image_clbk, queue_size = 10)
            rospy.loginfo("waiting for camera message...")
            rospy.wait_for_message(ros_image_input_topic, ROSImage, timeout=None)
            rospy.loginfo("... camera message arrived")
            self.ros_image_input = ROSImage
            
        self.keypoint_pub = rospy.Publisher(pub_out_keypoint_topic, KeypointImage, queue_size=10)
            
        if self.pub_out_images:
            self.image_pub = rospy.Publisher(pub_out_images_topic+"/compressed", ROSCompressedImage, queue_size=10)


    def __image_clbk(self, msg):
        
        self.ros_image_input = msg
        try:
            if self.camera_image_transport == "compressed":

                self.cv_image_input = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="rgb8")
            else :
                self.cv_image_input = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            
        except CvBridgeError as e:
            rospy.logerror(e)
            
        self.new_image = True
       
    def infer(self):
        
        if not self.new_image:
            #rospy.logwarn("no new image, publishing last results")
            if (len(self.out['scores']) == 0):
                #rospy.logwarn("no detection found at all (len is 0)")
                self.__pubROS(self.inference_stamp)
            else:
                self.__pubROS(self.inference_stamp, self.best_index, self.out['boxes'], self.out['scores'], self.out['labels'])
            return False
        
        with torch.no_grad():
            
            #tic = rospy.Time().now()
            #tic_py = time.time()
            self.out = self.model_helper.infer(self.cv_image_input)
            #self.out = non_max_suppression(out, 80, self.confidence_th, self.nms_th)
        
            #toc = rospy.Time().now()
            #toc_py = time.time()
            #rospy.loginfo ('Inference time: %s s', (toc-tic).to_sec())
            #rospy.loginfo ('Inference time py: %s s', toc_py-tic_py )
            #rospy.loginfo ('%s', toc_py-tic_py )

            #images[0] = images[0].detach().cpu()
        
        if (len(self.out['scores']) == 0):
            #rospy.logwarn("no detection found at all (len is 0)")
            self.__pubROS(self.inference_stamp)
            return False
        
        #IDK if the best box is always the first one, so lets the argmax
        self.best_index = torch.argmax(self.out['scores'])
        
        #show_image_with_boxes(img, self.out['boxes'][self.best_index], self.out['labels'][self.best_index])
        
        self.inference_stamp = rospy.Time.now()
        self.__pubROS(self.inference_stamp, self.best_index, self.out['boxes'], self.out['scores'], self.out['labels'])
        
        self.new_image = False
        
        return True
    
    def __pubROS(self, stamp, best_index=-1, box=None, score=None, label=None):
        
        if (best_index == -1):
            self.__pubKeypoint(stamp)
            
            if self.pub_out_images:
                self.__pubImageWithRectangle()

        else:
            self.__pubKeypoint(stamp, box[best_index], score[best_index], label[best_index])
            
            if self.pub_out_images:
                if self.pub_out_all_keypoints:
                    self.__pubImageWithAllRectangles(box, label)
                else:
                    self.__pubImageWithRectangle(box[best_index], score[best_index], label[best_index])
                
            

    def __pubImageWithRectangle(self, box=None, score=None, label=None):
        
        #first convert back to unit8
        self.cv_image_output = torchvision.transforms.functional.convert_image_dtype(
            self.model_helper.tensor_images[0].cpu(), torch.uint8).numpy().transpose([1,2,0])
        self.cv_image_output = cv2.cvtColor(self.cv_image_output, cv2.COLOR_BGR2RGB)
        
        if (not box == None) and (score > self.detection_confidence_threshold) :
            cv2.rectangle(self.cv_image_output, 
                          (round(box[0].item()), round(box[1].item())),
                          (round(box[2].item()), round(box[3].item())),
                          (255,0,0), 3)
        
        #if label:
            #cv2.putText(self.cv_image_output, str(label.item()), (round(box[0].item()), round(box[3].item()+10)), 
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        
        #cv2.imshow("test_boxes", self.cv_image_output)
        #cv2.waitKey()
        
        self.ros_image_output = self.bridge.cv2_to_compressed_imgmsg(self.cv_image_output)
        self.ros_image_output.header.seq = self.ros_image_input.header.seq
        self.ros_image_output.header.frame_id = self.ros_image_input.header.frame_id
        self.ros_image_output.header.stamp = rospy.Time.now()
        
        self.image_pub.publish(self.ros_image_output)
        
    def __pubImageWithAllRectangles(self, box=None, label=None):
        
        #first convert back to unit8
        self.cv_image_output = torchvision.transforms.functional.convert_image_dtype(
            self.model_helper.tensor_images[0].cpu(), torch.uint8).numpy().transpose([1,2,0])
        self.cv_image_output = cv2.cvtColor(self.cv_image_output, cv2.COLOR_BGR2RGB)
        
        if not box == None:
            i = 0
            for b in box:
                cv2.rectangle(self.cv_image_output, 
                              (round(b[0].item()), round(b[1].item())),
                              (round(b[2].item()), round(b[3].item())),
                              (255,0,0), 2)
        
                if not label == None:
                        cv2.putText(self.cv_image_output, str(label[i].item()), (round(b[0].item()), round(b[3].item()+10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                i = i+1
        
        #cv2.imshow("test_boxes", self.cv_image_output)
        #cv2.waitKey()
        
        self.ros_image_output = self.bridge.cv2_to_compressed_imgmsg(self.cv_image_output)
        self.ros_image_output.header.seq = self.ros_image_input.header.seq
        self.ros_image_output.header.frame_id = self.ros_image_input.header.frame_id
        self.ros_image_output.header.stamp = rospy.Time.now()
        
        self.image_pub.publish(self.ros_image_output)

            
    """
    box is tensor and may be still float, we round befor filling the msg
    """        
    def __pubKeypoint(self, stamp, box=None, score=None, label=None):
        
        msg = KeypointImage()
        msg.header.frame_id = self.ros_image_input.header.frame_id
        msg.header.seq = self.ros_image_input.header.seq
        msg.header.stamp = stamp
        
        if (not box == None) and (not score == None) and (not label == None):
        
            #box from model has format: [x_0, y_0, x_1, y_1]
            msg.x_pixel = round(box[0].item() + (box[2].item() - box[0].item())/2)
            msg.y_pixel = round(box[1].item() + (box[3].item()  - box[1].item())/2)
            msg.label = label
            msg.confidence = score
            
        else:
            msg.x_pixel = 0
            msg.y_pixel = 0
            msg.label = 0
            msg.confidence = 0
        
        self.keypoint_pub.publish(msg)
        

if __name__=="__main__":
    # Initialize node
    rospy.init_node("2D_tracking")
    
    rate_param = rospy.get_param('~rate', 5)

    # Define detector object
    dm = DetectorManager()

    rate = rospy.Rate(rate_param) # ROS Rate
    
    while not rospy.is_shutdown():
        new_infer = dm.infer()
        rate.sleep()
    

    
