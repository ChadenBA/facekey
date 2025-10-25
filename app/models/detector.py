# models/detector.py
import cv2
import numpy as np
import insightface

class FaceDetector:
    def __init__(self):
        # Load RetinaFace detector
        self.model = insightface.app.FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.model.prepare(ctx_id=0)

    def detect(self, image: np.ndarray):
        # Convert to RGB for InsightFace
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.model.get(image_rgb)
        return faces
