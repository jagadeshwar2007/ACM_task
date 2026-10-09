from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

house = {"longitude": -122.23, "latitude": 37.88, "housing_median_age": 41, "total_rooms": 880,
         "total_bedrooms": 129, "population": 322, "households": 126, "median_income": 8.3252,
         "ocean_proximity": "NEAR BAY"}


def test_home():
    assert client.get("/").status_code == 200


def test_predict():
    r = client.post("/predict", json=house)
    assert r.status_code == 200
    assert r.json()["predicted_price"] > 0


def test_bad_proximity():
    r = client.post("/predict", json={**house, "ocean_proximity": "MARS"})
    assert r.status_code == 422


def test_missing_field():
    bad = dict(house)
    del bad["median_income"]
    assert client.post("/predict", json=bad).status_code == 422
