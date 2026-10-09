from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# model is trained in task 6
model_path = Path(__file__).resolve().parent.parent / "task_06_housing_price_predictor" / "model" / "housing_model.joblib"
model = joblib.load(model_path)

app = FastAPI(title="Housing price API")

PROXIMITY = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]


class House(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str


@app.get("/")
def home():
    return {"message": "housing price api is running, go to /docs to try it"}


@app.post("/predict")
def predict(house: House):
    if house.ocean_proximity not in PROXIMITY:
        raise HTTPException(status_code=422, detail=f"ocean_proximity must be one of {PROXIMITY}")
    data = pd.DataFrame([house.model_dump()])
    price = model.predict(data)[0]
    return {"predicted_price": round(float(price), 2)}
