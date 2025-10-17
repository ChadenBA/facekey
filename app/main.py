from fastapi import FastAPI
from app.routes.face import router as face_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="FaceKey - Face Authentication API")
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
@app.get("/")
async def home():
    return {"message": "FaceKey API is running!"}
app.include_router(face_router, prefix="/face", tags=["Face Recognition"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
