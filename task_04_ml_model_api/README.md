# Task 4 — ML Model Web API (FastAPI)

A FastAPI microservice that loads the trained **scikit-learn** housing-price pipeline (from Task 6: imputation →
StandardScaler → one-hot → Random Forest) and serves **JSON predictions**.

## Run it

```bash
cd task_04_ml_model_api
pip install -r requirements.txt
uvicorn main:app --reload
```

Open **http://127.0.0.1:8000/docs** for the interactive Swagger UI (try requests from the browser).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/model-info` | Model params, test metrics, expected features, feature importances |
| POST | `/predict` | Predict one district |
| POST | `/predict-batch` | Predict up to 1000 districts in one call |

Inputs are validated with Pydantic (types and sensible ranges). An unknown `ocean_proximity` or a missing field
returns a clear **HTTP 422** error instead of a crash.

## Sample request / response

**Request**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "longitude": -122.23, "latitude": 37.88, "housing_median_age": 41,
        "total_rooms": 880, "total_bedrooms": 129, "population": 322,
        "households": 126, "median_income": 8.3252, "ocean_proximity": "NEAR BAY"
      }'
```

**Response**

```json
{"predicted_median_house_value": 431942.36, "currency": "USD"}
```

**Batch request** (`POST /predict-batch`) takes a JSON list of the same objects and returns a list of predictions, e.g.
`[{"predicted_median_house_value":431942.36,"currency":"USD"},{"predicted_median_house_value":75914.0,"currency":"USD"}]`

**Invalid input** (`"ocean_proximity": "MARS"`) →
```json
{"detail": "ocean_proximity must be one of ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'], got 'MARS'"}
```

## Tests

```bash
pytest -q        # 5 tests: health, single prediction, batch, bad category, missing field
```

## Files
- `main.py` — the API
- The trained pipeline is loaded from `../task_06_housing_price_predictor/model/` (run Task 6 first if you cloned without the model file)
- `test_api.py` — automated tests
- `requirements.txt` — scikit-learn is pinned to the version used for training (pickled models are version-sensitive)
