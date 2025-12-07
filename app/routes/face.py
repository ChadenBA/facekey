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
from app.models.antispoof import AntiSpoofing
from prometheus_client import Counter, start_http_server , Gauge
import requests
import base64
from datetime import datetime
from app.models.load_emotion_model import load_emotion_model
from PIL import Image
emotion_model = load_emotion_model("best_model_weights.pth", device="cpu")



router = APIRouter()




def log_spoof(image, spoof_score):
    # convert image to base64
    _, buffer = cv2.imencode('.jpg', image)
    img_b64 = base64.b64encode(buffer).decode('utf-8')

    doc = {
        "timestamp": datetime.utcnow().isoformat(),
        "spoof_score": float(spoof_score),
        "user_attempt": None,
        "image_preview": img_b64
    }

    try:
        res = requests.post("http://localhost:5000", json=doc)
        print(res.status_code, res.text)
    except Exception as e:
        print("Error sending to Logstash:", e)


# ----------------------------
# Initialize models
# ----------------------------
detector = FaceDetector()
embedder = ArcFaceEmbedder()
anti_spoof = AntiSpoofing()

THRESHOLD = 0.8
SPOOF_THRESHOLD = 3 # change as needed
# Start Prometheus metrics server
start_http_server(8001)  # Prometheus will scrape metrics here

SPOOF_COUNTER = Counter('spoof_attempts_total', 'Total spoof attempts')
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Load fine-tuning model
# ----------------------------
train_csv = "train.csv"
train_df = pd.read_csv(train_csv)

input_dim = train_df.shape[1] - 1
num_classes = len(train_df["id"].unique())

ft_model = ArcFaceFineTuner(
    input_dim=input_dim,
    embedding_dim=512,
    num_classes=num_classes
).to(device)

ft_model.load_state_dict(torch.load("insightface_finetuned.pt", map_location=device))
ft_model.eval()


# ----------------------------
# Helper functions
# ----------------------------
def crop_face(image, face):
    x1, y1, x2, y2 = map(int, face.bbox)
    x1 = max(0, x1)
    y1 = max(0, y1)
    return image[y1:y2, x1:x2]




def mask_eyes(face, image):
    if not hasattr(face, "landmark_2d_106"):
        return image
    lm = face.landmark_2d_106
    left_eye, right_eye = lm[36:42], lm[42:48]
    x_min = int(min(left_eye[:, 0].min(), right_eye[:, 0].min()))
    x_max = int(max(left_eye[:, 0].max(), right_eye[:, 0].max()))
    y_min = int(min(left_eye[:, 1].min(), right_eye[:, 1].min()))
    y_max = int(max(left_eye[:, 1].max(), right_eye[:, 1].max()))

    roi = image[y_min:y_max, x_min:x_max]
    if roi.size > 0:
        avg = cv2.mean(roi)[:3]
        image[y_min:y_max, x_min:x_max] = np.array(avg, dtype=np.uint8)
    return image


def preprocess_face(face, image):
    masked = mask_eyes(face, image.copy())
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)


def robust_embedding(face_embedding: np.ndarray) -> np.ndarray:
    emb_np = face_embedding.astype(np.float32)
    if emb_np.shape[0] != ft_model.fc.in_features:
        if emb_np.shape[0] < ft_model.fc.in_features:
            emb_np = np.pad(emb_np, (0, ft_model.fc.in_features - emb_np.shape[0]))
        else:
            emb_np = emb_np[:ft_model.fc.in_features]

    x = torch.from_numpy(emb_np).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = ft_model(x).cpu().numpy()[0]
        emb /= np.linalg.norm(emb)
    return emb



def encode_image_for_db(image, size=(64, 64)):
    """Convert a cv2 image to a base64 string for logging."""
    resized = cv2.resize(image, size)
    _, buffer = cv2.imencode('.jpg', resized)
    return base64.b64encode(buffer).decode('utf-8')





# ----------------------------
# Metrics for model training
# ----------------------------



# ----------------------------
# REGISTER
# ----------------------------
@router.post("/register")
async def register_face(name: str = Form(...), file: UploadFile = File(...)):
    try:
        content = await file.read()
        npimg = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if image is None:
            return {"error": "Invalid image"}

        faces = detector.detect(image)
        if not faces:
            return {"error": "No face detected"}

        face = faces[0]
        face_crop = cv2.resize(crop_face(image, face), (128, 128))

        spoof_score = anti_spoof.predict(face_crop)
        print("DEBUG spoof score:", spoof_score)
        if spoof_score < SPOOF_THRESHOLD:
            log_spoof(face_crop, spoof_score)
            SPOOF_COUNTER.inc()
            await db.spoofs.insert_one({
            "timestamp": datetime.utcnow(),
             "spoof_score": float(spoof_score),
             "image_preview": encode_image_for_db(face_crop),  # small thumbnail/base64
              "user_attempt": None
          })
            return {"error": "Spoof detected! Real face required."}

        processed = preprocess_face(face, image)
        faces_proc = detector.detect(processed)
        if not faces_proc:
            return {"error": "No face after preprocessing"}

        embedding = embedder.get_embedding(faces_proc[0])
        embedding_ft = robust_embedding(embedding)

        existing = await db.faces.find_one({"name": name})
        if existing:
            return {"error": f"User '{name}' already exists."}

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


# ----------------------------
# VERIFY with Anti-Spoofing (simplified)
# ----------------------------
@router.post("/verify")
async def verify_face(file: UploadFile = File(...)):
    try:
        content = await file.read()
        npimg = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if image is None:
            return {"error": "Invalid image"}

        faces = detector.detect(image)
        if not faces:
            return {"error": "No face detected"}

        face = faces[0]
        face_crop = cv2.resize(crop_face(image, face), (128, 128))

        # 1. anti spoof check
        spoof_score = anti_spoof.predict(face_crop)
        print("DEBUG spoof score:", spoof_score)

        if spoof_score < SPOOF_THRESHOLD:
            log_spoof(face_crop, spoof_score)
            SPOOF_COUNTER.inc()
            await db.spoofs.insert_one({
                "timestamp": datetime.utcnow(),
                "spoof_score": float(spoof_score),
                "image_preview": encode_image_for_db(face_crop),
                "user_attempt": None
            })
            print("⚠️ SPOOFING DETECTED — score:", spoof_score)
            return {"message": "⚠️ Spoof detected! Real face required."}

        # 2. embedding
        processed = preprocess_face(face, image)
        faces_proc = detector.detect(processed)
        if not faces_proc:
            return {"error": "No face after preprocessing"}

        new_emb = embedder.get_embedding(faces_proc[0])
        new_emb_ft = robust_embedding(new_emb)

        best_score = 0
        best_user = None

        cursor = db.faces.find({})
        async for user in cursor:
            for emb in user.get("embeddings", []):
                emb_ft = np.array(emb, dtype=np.float32)
                if emb_ft.shape != new_emb_ft.shape:
                    continue
                score = float(np.dot(new_emb_ft, emb_ft))
                if score > best_score:
                    best_score = score
                    best_user = user["name"]

        verified = best_score >= THRESHOLD

        # ----------------------------------------------------------------------
         # 3. Emotion recognition (si vérifié)
        emotion_result = None
        
        if verified:
            # Convertir OpenCV (BGR) -> PIL (RGB)
            pil_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            
            # ✅ Utiliser la nouvelle signature
            pred_idx, pred_label, probs_dict = emotion_model.predict(pil_img, device="cpu")
            
            emotion_result = {
                "emotion": pred_label,
                "confidence": probs_dict[pred_label],
              #  "all_probabilities": probs_dict
            }

        return {
            "verified": verified,
            "user": best_user if verified else None,
            "similarity": round(best_score, 4),
            "emotion": emotion_result  # Dict complet ou None
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
        return {"error": f"Verification failed: {str(e)}"}


