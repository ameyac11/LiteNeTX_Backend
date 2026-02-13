"""LiteNeTX API — serves FashionMNIST, CIFAR-10, and CIFAR-100 models."""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
import io
from typing import Dict
from pathlib import Path

from app.ml.LiteNeTX_Base_CNN_FashionMNIST import load_fashion_model
from app.ml.LiteNeTX_Base_CNN_C10 import load_cifar_model
from app.ml.LiteNeTX_Base_CNN_C100 import load_cifar100_model
from app.ml.predictor import predict_fashion, predict_cifar10, predict_cifar100

app = FastAPI(title="LiteNeTX API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://litenetx.in", "https://www.litenetx.in", "https://litenetx.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

VALID_MODEL_TYPES = {"fashion", "cifar", "cifar100"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@app.on_event("startup")
async def startup_event():
    """Load all three models on startup."""
    load_fashion_model()
    load_cifar_model()
    load_cifar100_model()


@app.get("/api/examples/list/{model_type}")
async def list_examples(model_type: str) -> Dict:
    """List available example images for a model."""
    if model_type not in VALID_MODEL_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid model type. Use one of: {VALID_MODEL_TYPES}")

    examples_dir = Path(__file__).parent.parent / "examples" / model_type
    if not examples_dir.exists():
        return {"model": model_type, "examples": []}

    image_files = sorted([f for f in examples_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS])

    examples = [
        {"filename": img.name, "url": f"/examples/{model_type}/{img.name}", "label": f"Example {i + 1}"}
        for i, img in enumerate(image_files)
    ]
    return {"model": model_type, "examples": examples}


@app.options("/examples/{model_type}/{filename}")
async def options_example_image(model_type: str, filename: str):
    return JSONResponse(content={})


@app.get("/examples/{model_type}/{filename}")
async def get_example_image(model_type: str, filename: str):
    """Serve example images with CORS headers."""
    if model_type not in VALID_MODEL_TYPES:
        raise HTTPException(status_code=400, detail="Invalid model type")

    file_path = Path(__file__).parent.parent / "examples" / model_type / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid file type")

    return FileResponse(
        file_path,
        media_type=f"image/{file_path.suffix[1:]}",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )


@app.get("/")
async def root() -> Dict:
    return {"status": "ok", "service": "LiteNeTX API", "version": "2.0"}


@app.get("/health")
async def health() -> Dict:
    return {
        "status": "ok",
        "models": {
            "fashion": True,
            "cifar": True,
            "cifar100": True
        }
    }


async def _handle_prediction(file: UploadFile, predict_fn):
    """Shared prediction handler for all endpoints."""
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    if not file.content_type or not file.content_type.startswith("image/"):
        return JSONResponse(status_code=415, content={"error": True, "message": "Invalid file type. Please upload an image."})

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception:
        return JSONResponse(status_code=400, content={"error": True, "message": "Corrupted or invalid image file"})

    try:
        return predict_fn(image)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": True, "message": f"Prediction failed: {str(e)}"})


@app.post("/predict/fashion")
async def predict_fashion_endpoint(file: UploadFile = File(...)) -> Dict:
    """Predict FashionMNIST class using LiteNeTX-1."""
    return await _handle_prediction(file, predict_fashion)


@app.post("/predict/cifar")
async def predict_cifar_endpoint(file: UploadFile = File(...)) -> Dict:
    """Predict CIFAR-10 class using LiteNeTX-2."""
    return await _handle_prediction(file, predict_cifar10)


@app.post("/predict/cifar100")
async def predict_cifar100_endpoint(file: UploadFile = File(...)) -> Dict:
    """Predict CIFAR-100 class using LiteNeTX-3."""
    return await _handle_prediction(file, predict_cifar100)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(status_code=500, content={"error": True, "message": "An unexpected error occurred"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
