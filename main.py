from ultralytics import YOLO
import yaml
import torch
from PIL import Image
import os
import cv2
import time


yaml_filename = './fire.yaml'

model = YOLO('yolov8n.yaml')
model = YOLO('yolov8n.pt')
model = YOLO('yolov8n.yaml').load('yolov8n.pt')


results = model.train(data= yaml_filename, epochs = 5, imgsz = 640)

# model = YOLO('runs/detect/train15/weights/best.pt')
# results = model.val(data=yaml_filename)

metrics = results.box
print(f"mAP@0.5: {metrics.map50}")
print(f"mAP@0.5:0.95: {metrics.map}")
print(f"Precision: {metrics.p}")
print(f"Recall: {metrics.r}")

print(f"\n\nResults: {results.results_dict}")
print(f"Maps: {results.maps}")
# print(f"Mean Results: {results.mean_results}") # This throws an error
