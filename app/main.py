"""Main FastAPI application."""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
from typing import Dict, List
from pathlib import Path
import os

from app.ml.fashion_model import load_fashion_model
from app.ml.cifar_model import load_cifar_model
from app.ml.predictor import predict_fashion, predict_cifar


# Application configuration
APP_NAME = "LiteNeTX API"
APP_VERSION = "1.0"


# Initialize FastAPI app
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://litenetx.in", "https://www.litenetx.in", "https://litenetx.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    load_fashion_model()
    load_cifar_model()


@app.get("/api/examples/list/{model_type}")
async def list_examples(model_type: str) -> Dict:
    """
    List available example images for a model.
    
    Args:
        model_type: Either 'fashion' or 'cifar'
        
    Returns:
        JSON with list of example images and their URLs
    """
    if model_type not in ['fashion', 'cifar']:
        raise HTTPException(status_code=400, detail="Invalid model type. Use 'fashion' or 'cifar'")
    
    examples_dir = Path(__file__).parent.parent / "examples" / model_type
    
    if not examples_dir.exists():
        return {
            "model": model_type,
            "examples": []
        }
    
    # Get all image files in the directory
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    image_files = [
        f for f in examples_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    # Sort by filename
    image_files.sort()
    
    # Build response with URLs
    examples = []
    for idx, img_file in enumerate(image_files):
        examples.append({
            "filename": img_file.name,
            "url": f"/examples/{model_type}/{img_file.name}",
            "label": f"Example {idx + 1}"
        })
    
    return {
        "model": model_type,
        "examples": examples
    }


@app.options("/examples/{model_type}/{filename}")
async def options_example_image(model_type: str, filename: str):
    """Handle CORS preflight requests for example images."""
    return JSONResponse(content={})


@app.get("/examples/{model_type}/{filename}")
async def get_example_image(model_type: str, filename: str):
    """
    Serve example images with proper CORS headers.
    
    Args:
        model_type: Either 'fashion' or 'cifar'
        filename: Name of the image file
        
    Returns:
        The image file with CORS headers
    """
    if model_type not in ['fashion', 'cifar']:
        raise HTTPException(status_code=400, detail="Invalid model type")
    
    examples_dir = Path(__file__).parent.parent / "examples" / model_type
    file_path = examples_dir / filename
    
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Verify it's actually an image file (security check)
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    if file_path.suffix.lower() not in image_extensions:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    # Return FileResponse with explicit CORS headers
    return FileResponse(
        file_path,
        media_type=f"image/{file_path.suffix[1:]}",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )


# Mount static files for example images (MUST come after API routes)
# Note: The endpoint above provides better CORS control
# examples_path = Path(__file__).parent.parent / "examples"
# if examples_path.exists():
#     app.mount("/examples", StaticFiles(directory=str(examples_path)), name="examples")


@app.get("/")
async def root() -> Dict:
    """Root endpoint."""
    return {
        "status": "ok",
        "service": "LiteNeTX API",
        "version": "1.0"
    }


@app.get("/health")
async def health() -> Dict:
    """Health check endpoint."""
    return {
        "status": "ok",
        "models": {
            "fashion": True,
            "cifar": True
        }
    }


@app.post("/predict/fashion")
async def predict_fashion_endpoint(file: UploadFile = File(...)) -> Dict:
    """
    Predict FashionMNIST class for uploaded image.
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON with model name, top1, and top3 predictions
    """
    try:
        # Validate file is provided
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            return JSONResponse(
                status_code=415,
                content={"error": True, "message": "Invalid file type. Please upload an image."}
            )
        
        # Read and open image
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": True, "message": "Corrupted or invalid image file"}
            )
        
        # Make prediction
        result = predict_fashion(image)
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": True, "message": f"Internal server error: {str(e)}"}
        )


@app.post("/predict/cifar")
async def predict_cifar_endpoint(file: UploadFile = File(...)) -> Dict:
    """
    Predict CIFAR-10 class for uploaded image.
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON with model name, top1, and top3 predictions
    """
    try:
        # Validate file is provided
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            return JSONResponse(
                status_code=415,
                content={"error": True, "message": "Invalid file type. Please upload an image."}
            )
        
        # Read and open image
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": True, "message": "Corrupted or invalid image file"}
            )
        
        # Make prediction
        result = predict_cifar(image)
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": True, "message": f"Internal server error: {str(e)}"}
        )


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Catch-all exception handler."""
    return JSONResponse(
        status_code=500,
        content={"error": True, "message": "An unexpected error occurred"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
