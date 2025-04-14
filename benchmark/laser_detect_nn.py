#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tori
with ideas taken from https://github.com/vvasilo/yolov3_pytorch_ros/blob/master/src/yolov3_pytorch_ros/detector.py
"""
#import sys
import os
import numpy as np
import time
import json
import sys

# Pytorch stuff
import torch
import torchvision.transforms

from pycocotools.coco import COCO

#Opencv stuff
import cv2


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
            self.model = torch.hub.load(yolo_path, 'custom', source='local', path=model_path, force_reload=True)

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
    
    cv_image_input = cv_image_output = np.zeros((100,100,3), np.uint8)
    new_image = False
    model_helper = None
    out = {'scores' : []}
    best_index = -1
    inference_stamp = None
    
    def __init__(self, model_path=None, 
                 model_name=None, 
                 yolo_path="ultralytics/yolov5", 
                 detection_confidence_threshold=0.55,
                 display=False):

        ### Input Params
        self.model_path = model_path
        self.model_name = model_name
        self.yolo_path = yolo_path

        ## Detection Params
        self.detection_confidence_threshold = detection_confidence_threshold

        self.display = display
        
        if (model_name.startswith('yolo')) :
            self.model_helper = YoloModel()
        
        else:
            self.model_helper = NoYoloModel()
        
        ############ PYTHORCH STUFF
        model_path_name = os.path.join(model_path, model_name)
        
        print(f"Using model {model_path_name}")
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("CUDA available, use GPU")
            self.model_helper.initialize(model_path_name, yolo_path, 'gpu')

        else:
            self.device = torch.device('cpu')
            print("CUDA not available, use CPU") 
            self.model_helper.initialize(model_path_name, yolo_path, 'cpu')
        
       
    def infer(self, cv_image_input):
        
        with torch.no_grad():
            
            tic_py = time.time()
            self.out = self.model_helper.infer(cv_image_input)
            #self.out = non_max_suppression(out, 80, self.confidence_th, self.nms_th)
            toc_py = time.time()
            infer_time = toc_py - tic_py
            print("inference time: ", infer_time)

            #images[0] = images[0].detach().cpu()
        
        if (len(self.out['scores']) == 0):
            return [], [], [], [], infer_time
        
        #IDK if the best box is always the first one, so lets the argmax
        best_index = torch.argmax(self.out['scores'])
        
        if self.display:
            im_with_keypoints = cv_image_input.copy()
            for point in self.out['boxes'].detach().cpu().numpy():
                # Draw the keypoint on the image
                cv2.rectangle(im_with_keypoints, 
                       (round(point[0]), round(point[1])), 
                       (round(point[2]), round(point[3])), 
                       (255, 0, 0), 
                       1)
                
            print(f'time: ' + str(infer_time))
            cv2.imshow('Original Image', cv_image_input)
            cv2.imshow('im_with_keypoints', im_with_keypoints)
            cv2.waitKey(0)

        #box from model has format: [x_0, y_0, x_1, y_1]
        return self.out['boxes'], self.out['scores'], self.out['labels'], best_index, infer_time
        

if __name__=="__main__":

    root_dir = "/home/tori/YOLO/"
    dataset_name = "laser_home"
    model_name = "yolov5l6_e200_b8_tvt302010_laser_v5"

    annotation = root_dir + dataset_name + "/annotations/instances_default.json"
    img_dir = root_dir + dataset_name + "/images/"
    coco = COCO(annotation)
    
    ######### RUN DETECTION 
    dm = DetectorManager(model_path=root_dir, 
                         model_name= model_name + ".pt", 
                         yolo_path=root_dir + "/yolov5", 
                         detection_confidence_threshold=0.55,
                         display=False)

    detections = []
    det_times = []
    mean_det_time = 0.0
    for img_id in coco.getImgIds():
        filename = coco.loadImgs(img_id)[0]['file_name']
        img = cv2.imread(os.path.join(img_dir,filename))
        yolo_bboxes, scores, labels, best_index, det_time = dm.infer(img)

        if len(yolo_bboxes) == 0:
            print(f"no detection found at all for {filename} (len is 0)")

        for i, bb in enumerate(yolo_bboxes):

            # print("type of bb: ", type(bb))
            # print("type of img_id: ", type(img_id))
            # print("type of labels: ", type(labels))
            # print("type of scores: ", type(scores))
            # print("type of detections: ", type(detections))

            bb_cpu = bb.detach().cpu().numpy().tolist()
            labels_cpu = labels.detach().cpu().numpy().tolist()
            scores_cpu = scores.detach().cpu().numpy().tolist()

            detections.append({
                "image_id": img_id,
                "category_id": 1,
                "bbox": [round(bb_cpu[0],2), round(bb_cpu[1],2), round(bb_cpu[2]-bb_cpu[0],2), round(bb_cpu[3]-bb_cpu[1],2)],
                #"score": 1.0, #others method have no score so 1 for everyone?
                "score": scores_cpu[i],
            })

        det_times.append({
            "image_id": img_id,
            "detect_time": det_time
        })
        mean_det_time += det_time

    cv2.destroyAllWindows()

    with open("results/" + dataset_name+"_"+model_name+"_detections.json", 'w') as f:
        json.dump(detections, f)   

    with open("results/" + dataset_name+"_"+model_name+"_detections_time.json", 'w') as f:
        json.dump(det_times, f)     
