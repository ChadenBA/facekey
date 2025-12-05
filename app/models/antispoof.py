# app/models/anti_spoofing.py
import onnxruntime
import cv2
import numpy as np
class AntiSpoofing:
    def __init__(self, model_path="./app/models/AntiSpoofing_bin_1.5_128.onnx"):
        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        # Input details
        self.input_name = self.session.get_inputs()[0].name
        self.height = 128
        self.width = 128

    def preprocess(self, face_image):
        """Resize + normalize"""
        img = cv2.resize(face_image, (self.width, self.height))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW
        img = np.expand_dims(img, axis=0)   # NCHW
        return img

    def predict(self, face_image):
        """
        Returns:
            score (float): 0 = spoof (fake), 1 = real
        """
        img = self.preprocess(face_image)
        ort_inputs = {self.input_name: img}
        output = self.session.run(None, ort_inputs)[0]

        # Model outputs a logit → apply softmax
        prob_real = float(output[0][1])  # index 1 = real face
        return prob_real
