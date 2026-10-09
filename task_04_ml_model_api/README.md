# Task 4: ML model web API

A FastAPI app that loads the housing price model from task 6 and returns predictions as JSON.

## How to run

```
cd task_04_ml_model_api
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open http://127.0.0.1:8000/docs to try it in the browser.

## Endpoints

- `GET /` - checks that the api is running
- `POST /predict` - send the house details, get the predicted price

## Sample request

```
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"longitude": -122.23, "latitude": 37.88, "housing_median_age": 41, "total_rooms": 880, "total_bedrooms": 129, "population": 322, "households": 126, "median_income": 8.3252, "ocean_proximity": "NEAR BAY"}'
```

## Sample response

```
{"predicted_price": 431942.36}
```

If `ocean_proximity` is not one of `<1H OCEAN`, `INLAND`, `ISLAND`, `NEAR BAY`, `NEAR OCEAN`, it returns a 422 error. A missing field also gives a 422 error.

## Tests

```
pytest
```
