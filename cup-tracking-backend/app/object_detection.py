from ultralytics import YOLO
import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, model_path='../models/yolov8n.pt'):
        self.model = YOLO('../models/yolov8n.pt')  # load the YOLOv8 model

    def detect_objects(self, image):
        results = self.model(image)
        return results

    def get_cup_bbox(self, results):
        bboxes = []
        for result in results[0].boxes:
            if result.cls == 41:  # class ID for cup in COCO dataset is 41
                bboxes.append(result.xyxy.cpu().numpy())
        return bboxes
