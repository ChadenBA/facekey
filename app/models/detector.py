import cv2
from insightface.app import FaceAnalysis
import numpy as np

class FaceDetector:
    def __init__(self):
        try:
            self.model = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.model.prepare(ctx_id=0, det_size=(640, 640))
        except Exception as e:
            print(f"[Warning] GPU load failed: {e}")
            self.model = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
            self.model.prepare(ctx_id=-1, det_size=(640, 640))

    def detect(self, image):
        if image.dtype != np.uint8:
            # Convert float64 or float32 to uint8
            image = np.clip(image, 0, 255).astype(np.uint8)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.model.get(image_rgb)
        return faces
