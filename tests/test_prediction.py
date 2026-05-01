from src.models.predict_model import load_model, predict_fare


def sample_trip_payload():
    return {
        # Numeric features
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

        # Binary features
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

        # Categorical features
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


def test_model_loads():
    model = load_model()
    assert model is not None


def test_prediction_returns_float():
    model = load_model()
    prediction = predict_fare(sample_trip_payload(), model=model)

    assert isinstance(prediction, float)


def test_prediction_is_positive():
    model = load_model()
    prediction = predict_fare(sample_trip_payload(), model=model)

    assert prediction > 0