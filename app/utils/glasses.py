from ultralytics import YOLO
import cv2
import numpy as np
import torch

# Load YOLOv8 model for glasses classification
model = YOLO("yolov8n-cls.pt")  # replace with your trained weights path

def detect_glasses_cnn(face_crop: np.ndarray, threshold: float = 0.3) -> bool:
    """
    Detect if glasses are present in a cropped face using YOLOv8.
    
    Args:
        face_crop (np.ndarray): BGR face crop
        threshold (float): probability threshold to consider glasses detected

    Returns:
        bool: True if glasses detected, else False
    """
    if face_crop is None or face_crop.size == 0:
        return False

    # Preprocess image
    img = cv2.resize(face_crop, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # YOLO prediction
    results = model.predict(img_tensor, imgsz=128, verbose=False)
    probs_obj = results[0].probs
    if probs_obj is None:
        return False

    top_class = probs_obj.top1
    top_conf = float(probs_obj.top1conf)

    # Adjust according to your trained labels
    glasses_class = 1  # 0 or 1 depending on your training
    return top_class == glasses_class and top_conf > threshold
