from __future__ import annotations

import asyncio
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from pydantic import BaseModel
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import models

from .inference import build_model_for_inference, get_llm_recommendations
from .preprocessing.pipeline import PreprocessingConfig, preprocess_image_for_model
from .preprocessing.segmentation import SegmentationConfig

app = FastAPI(
    title="Crop Disease Detection API",
    version="1.0.0",
    description="Made by team ClimaCrop.",
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
_model = None
_idx_to_class = None
_device = None
_preproc_cfg = None


def load_model(config_path: str = "config.yaml"):
    """Load the trained model and configuration."""
    global _model, _idx_to_class, _device, _preproc_cfg

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    seg_cfg_raw = cfg.get("segmentation", {})
    train_cfg = cfg["training"]

    seg_cfg = SegmentationConfig(
        hsv_lower=tuple(seg_cfg_raw.get("hsv_lower", [25, 40, 40])),
        hsv_upper=tuple(seg_cfg_raw.get("hsv_upper", [90, 255, 255])),
        kernel_size=seg_cfg_raw.get("kernel_size", 5),
        background_blur=seg_cfg_raw.get("background_blur", True),
    )

    _preproc_cfg = PreprocessingConfig(
        img_size=data_cfg["img_size"],
        segmentation=seg_cfg,
        save_debug=False,
    )

    models_dir = Path(train_cfg.get("output_dir", "models"))
    best_model_path = models_dir / "best_model.pt"
    mapping_path = models_dir / "class_mapping.json"

    if not best_model_path.is_file():
        raise FileNotFoundError(f"Best model not found at {best_model_path}")
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Class mapping not found at {mapping_path}")

    checkpoint = torch.load(best_model_path, map_location="cpu")
    backbone = checkpoint["backbone"]
    num_classes = checkpoint["num_classes"]
    state_dict = checkpoint["model_state_dict"]

    with mapping_path.open("r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    _idx_to_class = {v: k for k, v in class_to_idx.items()}

    _model = build_model_for_inference(backbone, num_classes, state_dict)
    device_str = train_cfg.get("device", "cuda")
    _device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    _model.to(_device)
    _model.eval()


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        load_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend HTML."""
    html_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Crop Disease Detection API</h1><p>Frontend not found. Please ensure frontend/index.html exists.</p>")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": _model is not None}


class RecommendationRequest(BaseModel):
    disease_name: str
    llm_model: str = "llama3.1:8b"


@app.post("/api/recommendations")
async def get_recommendations_endpoint(request: RecommendationRequest):
    """
    Get treatment recommendations for a disease.
    This endpoint can be called separately after prediction for faster response.
    
    Body:
        disease_name: The disease name to get recommendations for
        llm_model: Ollama model to use (optional, default: llama3.1:8b)
    """
    try:
        loop = asyncio.get_event_loop()
        recommendations = await loop.run_in_executor(
            None,
            get_llm_recommendations,
            request.disease_name,
            request.llm_model
        )
        return JSONResponse(content={
            "disease_name": request.disease_name,
            "recommendations": recommendations,
        })
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    include_recommendations: bool = True,
    llm_model: str = "llama3.1:8b",
):
    """
    Predict disease from uploaded image.
    
    Args:
        file: Image file to analyze
        include_recommendations: Whether to include LLM recommendations
        llm_model: Ollama model to use for recommendations
    """
    if _model is None or _idx_to_class is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Read and validate image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Save temporarily for preprocessing (required by preprocessing pipeline)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            image.save(tmp_file.name, "JPEG", quality=95)
            tmp_path = tmp_file.name

        try:
            # Preprocess image
            img_chw = preprocess_image_for_model(tmp_path, _preproc_cfg, debug_name=None)
            
            if img_chw is None:
                raise HTTPException(status_code=400, detail="Failed to preprocess image")

            # Run inference (fast - this is what works in terminal)
            x = torch.from_numpy(img_chw).unsqueeze(0).float().to(_device)
            with torch.no_grad():
                logits = _model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            pred_idx = int(np.argmax(probs))
            pred_class = _idx_to_class[pred_idx]
            confidence = float(probs[pred_idx])

            # Get top 3 predictions
            top_indices = np.argsort(probs)[::-1][:3]
            top_predictions = [
                {
                    "class": _idx_to_class[int(idx)],
                    "confidence": float(probs[int(idx)]),
                }
                for idx in top_indices
            ]

            result = {
                "predicted_class": pred_class,
                "confidence": confidence,
                "top_predictions": top_predictions,
            }

            # Get LLM recommendations if requested (this is the slow part)
            # Run in executor to avoid blocking, but still wait for it
            if include_recommendations:
                try:
                    loop = asyncio.get_event_loop()
                    recommendations = await loop.run_in_executor(
                        None, 
                        get_llm_recommendations, 
                        pred_class, 
                        llm_model
                    )
                    result["recommendations"] = recommendations
                except Exception as e:
                    result["recommendations"] = f"Error generating recommendations: {str(e)}"
                    result["recommendations_error"] = True
            else:
                # If recommendations not requested, return immediately
                result["recommendations"] = None
                result["recommendations_pending"] = True

            return JSONResponse(content=result)
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


# Mount static files if frontend directory exists
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
    
    # Serve CSS and JS files
    @app.get("/styles.css")
    async def get_css():
        css_path = frontend_dir / "styles.css"
        if css_path.exists():
            with open(css_path, "r", encoding="utf-8") as f:
                return Response(content=f.read(), media_type="text/css")
        raise HTTPException(status_code=404, detail="CSS file not found")
    
    @app.get("/script.js")
    async def get_js():
        js_path = frontend_dir / "script.js"
        if js_path.exists():
            with open(js_path, "r", encoding="utf-8") as f:
                return Response(content=f.read(), media_type="application/javascript")
        raise HTTPException(status_code=404, detail="JS file not found")
