import numpy as np

class ArcFaceEmbedder:
    def get_embedding(self, face):
        """
        Returns L2-normalized embedding as np.ndarray
        """
        emb = np.array(face.embedding, dtype=np.float32)
        emb /= np.linalg.norm(emb)
        return emb
