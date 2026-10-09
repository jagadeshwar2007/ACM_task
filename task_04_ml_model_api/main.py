"""Task 4 — Serving the housing-price model as a JSON API (FastAPI).

Run:   uvicorn main:app --reload
Docs:  http://127.0.0.1:8000/docs   (interactive Swagger UI)
"""
import json
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
# The trained pipeline is produced by Task 6 and shared (one copy in the repo instead of three).
MODEL_DIR = BASE.parent / "task_06_housing_price_predictor" / "model"
if not MODEL_DIR.exists():  # fallback: a local ./model folder if this folder is used on its own
    MODEL_DIR = BASE / "model"
MODEL_PATH = MODEL_DIR / "housing_model.joblib"
META_PATH = MODEL_DIR / "model_metadata.json"

model = joblib.load(MODEL_PATH)
metadata = json.loads(META_PATH.read_text())
VALID_PROXIMITY = metadata["ocean_proximity_values"]

app = FastAPI(
    title="California Housing Price API",
    description="Predicts the median house value of a California district with a Random Forest pipeline.",
    version="1.0.0",
)


class HouseFeatures(BaseModel):
    longitude: float = Field(..., ge=-125, le=-114, examples=[-122.23])
    latitude: float = Field(..., ge=32, le=42.5, examples=[37.88])
    housing_median_age: float = Field(..., ge=0, le=100, examples=[41])
    total_rooms: float = Field(..., gt=0, examples=[880])
    total_bedrooms: float = Field(..., gt=0, examples=[129])
    population: float = Field(..., gt=0, examples=[322])
    households: float = Field(..., gt=0, examples=[126])
    median_income: float = Field(..., gt=0, le=20, description="In tens of thousands of USD", examples=[8.3252])
    ocean_proximity: str = Field(..., examples=["NEAR BAY"])


class Prediction(BaseModel):
    predicted_median_house_value: float
    currency: str = "USD"


def _to_frame(items: List[HouseFeatures]) -> pd.DataFrame:
    for it in items:
        if it.ocean_proximity not in VALID_PROXIMITY:
            raise HTTPException(
                status_code=422,
                detail=f"ocean_proximity must be one of {VALID_PROXIMITY}, got '{it.ocean_proximity}'",
            )
    return pd.DataFrame([it.model_dump() for it in items])


@app.get("/")
def root():
    return {"message": "California Housing Price API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok", "model": metadata["model"]}


@app.get("/model-info")
def model_info():
    return {
        "params": metadata["params"],
        "test_metrics": metadata["test_metrics"],
        "features": metadata["features"],
        "ocean_proximity_values": VALID_PROXIMITY,
        "feature_importances": metadata["feature_importances"],
    }


@app.post("/predict", response_model=Prediction)
def predict(features: HouseFeatures):
    value = float(model.predict(_to_frame([features]))[0])
    return Prediction(predicted_median_house_value=round(value, 2))


@app.post("/predict-batch", response_model=List[Prediction])
def predict_batch(items: List[HouseFeatures]):
    if not items:
        raise HTTPException(status_code=422, detail="Send at least one record.")
    if len(items) > 1000:
        raise HTTPException(status_code=422, detail="Max 1000 records per request.")
    preds = model.predict(_to_frame(items))
    return [Prediction(predicted_median_house_value=round(float(p), 2)) for p in preds]
