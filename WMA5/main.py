# !git clone https://github.com/ultralytics/yolov5
# !pip install -r yolov5/requirements.txt

# Import necessary libraries
import torch
from yolov5 import train, detect

# Define the dataset path
data_path = 'yolov5/data/coco128.yaml'

# Train YOLOv5 model with random weights
train.run(imgsz=640, batch_size=16, epochs=50, data=data_path, cfg='yolov5m.yaml')

# Train YOLOv5 model with pre-trained weights
train.run(imgsz=640, batch_size=16, epochs=50, data=data_path, weights='yolov5m.pt')

# Define the path to the sample video
video_path = 'TestVideo.mp4'


# Perform detection using the model trained with random weights
detect.run(weights='yolov5/runs/train/exp/weights/best.pt', source=video_path, imgsz=(640,640))

# Perform detection using the model trained with pre-trained weights
detect.run(weights='yolov5/runs/train/exp2/weights/best.pt', source=video_path, imgsz=(640,640))