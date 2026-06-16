import os
import shutil
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from model import get_model
from predict import predict
from gradcam import generate_gradcam


app = FastAPI(title="Deepfake Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://deepfake-detector-iota.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
HEATMAP_DIR = "heatmaps"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)

app.mount("/heatmaps", StaticFiles(directory=HEATMAP_DIR), name="heatmaps")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading model on {device}...")
MODEL_WEIGHTS = "weights/best_model_v3.pth"
weights = MODEL_WEIGHTS if os.path.exists(MODEL_WEIGHTS) else None
model = get_model(device=device, weights_path=weights)
print("Model loaded successfully.")


@app.get("/")
async def root():
    return {
        "message": "Deepfake Detection API is running",
        "device": str(device),
        "status": "ready"
    }


@app.post("/detect")
async def detect_deepfake(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = predict(file_path, model, device=str(device))
        heatmap_path = generate_gradcam(
            model, file_path,
            output_dir=HEATMAP_DIR, device=str(device)
        )
        heatmap_url = f"/heatmaps/{os.path.basename(heatmap_path)}"

        return JSONResponse({
            "success": True,
            "prediction": result['label'],
            "confidence": result['confidence'],
            "probabilities": result['probabilities'],
            "heatmap_url": heatmap_url
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": str(device)}