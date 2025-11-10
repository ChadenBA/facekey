# app/routes/face.py
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.encoders import jsonable_encoder
from app.models.detector import FaceDetector
from app.models.embedder import ArcFaceEmbedder
from app.database.mongo import db
import numpy as np
import cv2
import traceback
import torch
import pandas as pd
from app.utils.fine_tune_insightface import ArcFaceFineTuner

router = APIRouter()

# ----------------------------
# Initialize models
# ----------------------------
detector = FaceDetector()
embedder = ArcFaceEmbedder()
THRESHOLD = 0.8  # cosine similarity threshold

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Determine input_dim and num_classes from training CSV ---
train_csv = "train.csv"
train_df = pd.read_csv(train_csv)
input_dim = train_df.shape[1] - 1  # number of embedding columns
num_classes = len(train_df["id"].unique())

# --- Load fine-tuned ArcFace model ---
ft_model = ArcFaceFineTuner(input_dim=input_dim, embedding_dim=512, num_classes=num_classes).to(device)
ft_model.load_state_dict(torch.load("insightface_finetuned.pt", map_location=device))
ft_model.eval()

# ----------------------------
# Helper functions
# ----------------------------
def mask_eyes(face, image):
    if not hasattr(face, "landmark_2d_106"):
        return image
    lm = face.landmark_2d_106
    left_eye, right_eye = lm[36:42], lm[42:48]
    x_min = int(min(left_eye[:,0].min(), right_eye[:,0].min()))
    x_max = int(max(left_eye[:,0].max(), right_eye[:,0].max()))
    y_min = int(min(left_eye[:,1].min(), right_eye[:,1].min()))
    y_max = int(max(left_eye[:,1].max(), right_eye[:,1].max()))
    roi = image[y_min:y_max, x_min:x_max]
    if roi.size > 0:
        avg_color = cv2.mean(roi)[:3]
        image[y_min:y_max, x_min:x_max] = np.array(avg_color, dtype=np.uint8)
    return image

def preprocess_face(face, image):
    masked = mask_eyes(face, image.copy())
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized = clahe.apply(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

def robust_embedding(face_embedding: np.ndarray) -> np.ndarray:
    """Convert original embedding to fine-tuned robust embedding"""
    # Make sure the embedding has the same size as model.fc.in_features
    emb_np = face_embedding.astype(np.float32)
    if emb_np.shape[0] != ft_model.fc.in_features:
        # If needed, pad or trim to match fc.in_features
        if emb_np.shape[0] < ft_model.fc.in_features:
            # pad with zeros
            emb_np = np.pad(emb_np, (0, ft_model.fc.in_features - emb_np.shape[0]))
        else:
            # truncate extra dimensions
            emb_np = emb_np[:ft_model.fc.in_features]

    x = torch.from_numpy(emb_np).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = ft_model(x).cpu().numpy()[0]
        emb /= np.linalg.norm(emb)
    return emb

# ----------------------------
# API Endpoints
# ----------------------------
@router.post("/register")
async def register_face(name: str = Form(...), file: UploadFile = File(...)):
    try:
        existing_user = await db.faces.find_one({"name": name})
        if existing_user:
            return {"error": f"User '{name}' already registered."}

        content = await file.read()
        npimg = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if image is None:
            return {"error": "Invalid image format"}

        faces = detector.detect(image)
        if not faces:
            return {"error": "No face detected"}

        face = faces[0]
        processed = preprocess_face(face, image)
        faces_proc = detector.detect(processed)
        if not faces_proc:
            return {"error": "No face after preprocessing"}

        embedding = embedder.get_embedding(faces_proc[0])
        embedding_ft = robust_embedding(embedding)

        await db.faces.insert_one({
            "name": name,
            "embeddings": [embedding_ft.tolist()]
        })

        return jsonable_encoder({
            "message": f"Face registered successfully for '{name}'"
        })

    except Exception as e:
        traceback.print_exc()
        return {"error": f"Registration failed: {str(e)}"}

@router.post("/verify")
async def verify_face(file: UploadFile = File(...)):
    try:
        content = await file.read()
        npimg = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if image is None:
            return {"error": "Invalid image format"}

        faces = detector.detect(image)
        if not faces:
            return {"error": "No face detected"}

        face = faces[0]
        processed = preprocess_face(face, image)
        faces_proc = detector.detect(processed)
        if not faces_proc:
            return {"error": "No face after preprocessing"}

        new_emb = embedder.get_embedding(faces_proc[0])
        new_emb_ft = robust_embedding(new_emb)

        # Compare with DB embeddings
        best_user = None
        best_score = 0.0

        # Use async cursor properly
        cursor = db.faces.find({})
        async for user in cursor:
            embeddings = user.get("embeddings", [])
            if not isinstance(embeddings, list):
                continue  # skip invalid records

            for emb in embeddings:
                try:
                    emb_ft = np.array(emb, dtype=np.float32)
                    if emb_ft.shape != new_emb_ft.shape:
                        continue  # skip mismatched embeddings
                    score = float(np.dot(new_emb_ft, emb_ft))
                except Exception:
                    continue  # skip invalid embeddings

                if score > best_score:
                    best_score = score
                    best_user = user.get("name")

        verified = best_score >= THRESHOLD
        return {
            "verified": bool(verified),
            "user": best_user if verified else None,
            "similarity": round(best_score, 4)
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": f"Verification failed: {str(e)}"}
