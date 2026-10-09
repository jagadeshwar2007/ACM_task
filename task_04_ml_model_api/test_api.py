from fastapi.testclient import TestClient
from main import app

client = TestClient(app)
SAMPLE = {"longitude": -122.23, "latitude": 37.88, "housing_median_age": 41, "total_rooms": 880,
          "total_bedrooms": 129, "population": 322, "households": 126, "median_income": 8.3252,
          "ocean_proximity": "NEAR BAY"}


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_predict():
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    assert 100_000 < r.json()["predicted_median_house_value"] < 600_000


def test_batch():
    r = client.post("/predict-batch", json=[SAMPLE, {**SAMPLE, "median_income": 2.0, "ocean_proximity": "INLAND"}])
    assert r.status_code == 200
    a, b = [x["predicted_median_house_value"] for x in r.json()]
    assert a > b  # high-income coastal district costs more than low-income inland one


def test_bad_category():
    assert client.post("/predict", json={**SAMPLE, "ocean_proximity": "MARS"}).status_code == 422


def test_missing_field():
    bad = dict(SAMPLE); bad.pop("median_income")
    assert client.post("/predict", json=bad).status_code == 422
