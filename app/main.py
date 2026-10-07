"""FastAPI service: GET /, GET /health, POST /predict."""
import io
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError

from app.schemas import HealthResponse, PredictionResponse
from src import config
from src.inference import (CheckpointNotFoundError, ClassMappingError, Predictor,
                           checkpoint_path)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}
MAX_BYTES = 10 * 1024 * 1024
MODEL_NAMES = config.ARCHES

app = FastAPI(title="Indian Traffic Sign Reader")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
_predictors: dict[str, Predictor] = {}


def _path_for(model: str) -> Path:
    override = os.environ.get(f"ITSR_{model.upper()}_CHECKPOINT")
    return Path(override) if override else checkpoint_path(model)


def get_predictor(model: str) -> Predictor:
    if model not in _predictors:
        _predictors[model] = Predictor(_path_for(model))
    return _predictors[model]


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"models": MODEL_NAMES})


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", models={m: _path_for(m).exists() for m in MODEL_NAMES})


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), model: str = Form("resnet18"), top_k: int = Form(3)):
    if model not in MODEL_NAMES:
        raise HTTPException(400, f"Invalid model {model!r}; choose one of {list(MODEL_NAMES)}.")
    if not 1 <= top_k <= 10:
        raise HTTPException(400, "top_k must be between 1 and 10.")
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"Unsupported content type {file.content_type!r}; "
                                 f"allowed: {sorted(ALLOWED_TYPES)}.")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image too large (limit 10 MB).")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise HTTPException(400, "File is not a valid, readable image.")
    try:
        predictor = get_predictor(model)
    except CheckpointNotFoundError as e:
        raise HTTPException(503, str(e))
    except ClassMappingError as e:
        raise HTTPException(500, str(e))
    p = predictor.predict(img, top_k)
    return PredictionResponse(model=model, prediction=p.label, confidence=p.probability,
                              calibrated_confidence=p.calibrated_probability, top_k=p.top_k,
                              latency_ms=p.latency_ms)
