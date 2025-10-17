from deepface import DeepFace
import cv2

class ArcFaceEmbedder:
    def get_embedding(self, face):
        # face = BGR image
        cv2.imwrite("temp.jpg", face)  # DeepFace uses file path
        embedding = DeepFace.represent(img_path="temp.jpg", model_name="Facenet512", detector_backend="mtcnn")
        return embedding[0]["embedding"]
