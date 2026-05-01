from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def sample_trip_payload():
    return {
        "PICKUP_HOUR": 14,
        "DAY_OF_WEEK": 2,
        "MONTH": 5,
        "YEAR": 2025,
        "PASSENGER_COUNT": 1,
        "TRIP_DISTANCE": 3.2,
        "TIME_OF_DAY": 2,
        "WEEK_OF_YEAR": 18,
        "QUARTER": 2,
        "DIST_X_RUSH": 0.0,
        "IS_RUSH_HOUR": 0,
        "IS_WEEKEND": 0,
        "IS_LATE_NIGHT": 0,
        "IS_AIRPORT_PU": 0,
        "IS_AIRPORT_DO": 0,
        "IS_AIRPORT_TRIP": 0,
        "SAME_BOROUGH": 1,
        "PU_IS_MANHATTAN": 1,
        "DO_IS_MANHATTAN": 1,
        "RUSH_AIRPORT": 0,
        "LATE_NIGHT_WEEKEND": 0,
        "SERVICE_TYPE": "yellow",
        "PU_BOROUGH": "Manhattan",
        "DO_BOROUGH": "Manhattan",
        "STORE_AND_FWD_FLAG": "N",
        "VENDOR_ID": 2,
        "PU_LOCATION_ID": 237,
        "DO_LOCATION_ID": 161,
        "RATE_CODE_ID": 1,
        "PAYMENT_TYPE": 1,
        "TRIP_TYPE": 1
    }


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_endpoint():
    response = client.post(
        "/predict",
        json=sample_trip_payload()
    )

    assert response.status_code == 200

    data = response.json()

    assert "estimated_fare_amount" in data
    assert "currency" in data
    assert "model" in data

    assert isinstance(data["estimated_fare_amount"], float)
    assert data["estimated_fare_amount"] > 0
    assert data["currency"] == "USD"