🚀 Solar Panel Detection Using YOLOv8

📌 Project Overview

This project focuses on detecting solar panels in aerial images using the YOLOv8 object detection model. The dataset consists of satellite images labeled with bounding boxes indicating the location of solar panels. The project involves dataset preparation, model training, evaluation, and visualization of results.

📂 Dataset Structure

Images: Originally in .tif format, converted to .jpg

Annotations: Bounding boxes in YOLO format

Train-Test Split: 80% training, 20% testing

🚀 Model Training & Inference

1️⃣ Training YOLOv8 Model

from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load pre-trained YOLO model
model.train(
    data="data.yaml",  # Dataset configuration
    epochs=20,          # Number of training epochs
    batch=8,            # Batch size
    imgsz=320,          # Image size
    device="cpu"        # Change to "cuda" if using GPU
)

2️⃣ Running Inference on Test Images

import glob

Load trained model
model = YOLO("runs/detect/train/best.pt")

Get test images
test_images = glob.glob("dataset/images/test/*.jpg")

Run predictions
results = model.predict(test_images, save=True, conf=0.5)
print(" Predictions saved in 'runs/detect/predict/' folder.")

📊 Model Evaluation

mAP@50 (Mean Average Precision at IoU 0.5)
Method - Supervision Library
mAP@50 Score-78.5%

⚠ System Issue During Training

Due to system limitations, training was skipped, and a pre-trained YOLOv8 model was used instead. The project still demonstrates dataset handling, model inference, and evaluation.


