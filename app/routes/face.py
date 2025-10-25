# routes/face.py
from fastapi import APIRouter, UploadFile, File, Form
from app.models.detector import FaceDetector
from app.models.embedder import ArcFaceEmbedder
from app.database.mongo import db
import numpy as np
import cv2

router = APIRouter()
detector = FaceDetector()
embedder = ArcFaceEmbedder()

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
        embedding = embedder.get_embedding(face)

        await db.faces.insert_one({"name": name, "embedding": embedding})
        return {"message": f"Face registered for {name}"}
    except Exception as e:
        return {"error": str(e)}


@router.post("/verify")
async def verify_face(file: UploadFile = File(...)):
    try:
        content = await file.read()
        npimg = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        faces = detector.detect(image)
        if not faces:
            return {"error": "No face detected"}

        new_emb = np.array(embedder.get_embedding(faces[0]))

        async for user in db.faces.find():
            stored_emb = np.array(user["embedding"])
            cos_sim = np.dot(new_emb, stored_emb) / (np.linalg.norm(new_emb) * np.linalg.norm(stored_emb))
            if cos_sim > 0.8:
                return {"verified": True, "user": user["name"], "similarity": float(cos_sim)}

        return {"verified": False, "message": "No match found"}
    except Exception as e:
        return {"error": str(e)}
