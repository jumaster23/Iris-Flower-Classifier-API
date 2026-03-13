"""
main.py — FastAPI server exposing the Iris classification model via REST API.

Endpoints:
    GET  /                     — Health check
    GET  /info                 — Model metadata
    GET  /models               — List available models
    POST /predict              — Single prediction (protected)
    POST /predict/batch        — Batch predictions (protected)
    POST /predict/{model_name} — Predict with specific model (protected)
"""

import json
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from auth import verify_api_key
from inference import IrisClassifier, load_classifier, AVAILABLE_MODELS
from settings import HOST, METADATA_PATH, MODEL_FILENAME, PORT, SAVED_MODELS_DIR


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

class PredictRequest(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm", json_schema_extra={"example": 5.1})
    sepal_width:  float = Field(..., gt=0, description="Sepal width in cm",  json_schema_extra={"example": 3.5})
    petal_length: float = Field(..., gt=0, description="Petal length in cm", json_schema_extra={"example": 1.4})
    petal_width:  float = Field(..., gt=0, description="Petal width in cm",  json_schema_extra={"example": 0.2})


class PredictResponse(BaseModel):
    predicted_class: str
    class_index: int
    probabilities: dict[str, float]
    input: dict[str, float]


class BatchPredictRequest(BaseModel):
    samples: list[PredictRequest] = Field(..., min_length=1, max_length=100)


class BatchPredictResponse(BaseModel):
    count: int
    predictions: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model: str
    ready: bool


# --------------------------------------------------------------------------- #
# App lifecycle
# --------------------------------------------------------------------------- #

classifier: IrisClassifier | None = None
metadata: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier, metadata

    print(f"Loading model: {MODEL_FILENAME}")
    model_path = SAVED_MODELS_DIR / MODEL_FILENAME

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run `uv run train.py` first."
        )

    classifier = IrisClassifier(model_path=model_path)

    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            metadata = json.load(f)

    print("Model loaded. Server ready.")
    yield
    print("Shutting down.")


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="Iris Flower Classifier API",
    description="Predicts iris species (setosa, versicolor, virginica) from petal/sepal measurements.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Routes — públicas
# --------------------------------------------------------------------------- #

@app.get("/", response_model=HealthResponse, tags=["Health"])
def health_check():
    return HealthResponse(status="ok", model=MODEL_FILENAME, ready=classifier is not None)


@app.get("/info", tags=["Model"])
def model_info():
    if not metadata:
        return {"message": "No metadata found. Run train.py first."}
    return metadata


@app.get("/models", tags=["Model"])
def list_models():
    return {"available_models": AVAILABLE_MODELS}


# --------------------------------------------------------------------------- #
# Routes — protegidas con API Key
# --------------------------------------------------------------------------- #

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"],
          dependencies=[Depends(verify_api_key)])
def predict(body: PredictRequest):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    features = [body.sepal_length, body.sepal_width, body.petal_length, body.petal_width]

    try:
        result = classifier.predict(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return PredictResponse(**result, input=body.model_dump())


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"],
          dependencies=[Depends(verify_api_key)])
def predict_batch(body: BatchPredictRequest):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    predictions = []
    for sample in body.samples:
        features = [sample.sepal_length, sample.sepal_width, sample.petal_length, sample.petal_width]
        try:
            result = classifier.predict(features)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
        predictions.append(PredictResponse(**result, input=sample.model_dump()))

    return BatchPredictResponse(count=len(predictions), predictions=predictions)


@app.post("/predict/{model_name}", response_model=PredictResponse, tags=["Prediction"],
          dependencies=[Depends(verify_api_key)])
def predict_with_model(model_name: str, body: PredictRequest):
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {AVAILABLE_MODELS}"
        )

    try:
        clf = load_classifier(model_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    features = [body.sepal_length, body.sepal_width, body.petal_length, body.petal_width]

    try:
        result = clf.predict(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return PredictResponse(**result, input=body.model_dump())


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)