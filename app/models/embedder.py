# models/embedder.py
import numpy as np

class ArcFaceEmbedder:
    def get_embedding(self, face):
        # face is a Face object from insightface
        return face.embedding.tolist()
