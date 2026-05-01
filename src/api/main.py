from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.predict_model import load_model, predict_fare


app = FastAPI(
    title="NYC Taxi Fare Prediction API",
    description="API for predicting NYC taxi fare amount using a trained LightGBM model.",
    version="1.0.0"
)

# Load model once when the API starts
model = load_model()


class TripInput(BaseModel):
    # Numeric features
    PICKUP_HOUR: int = Field(..., ge=0, le=23)
    DAY_OF_WEEK: int = Field(..., ge=0, le=7)
    MONTH: int = Field(..., ge=1, le=12)
    YEAR: int = Field(..., ge=2015, le=2030)
    PASSENGER_COUNT: int = Field(..., ge=1, le=6)
    TRIP_DISTANCE: float = Field(..., ge=0.1, le=150.0)
    TIME_OF_DAY: int = Field(..., ge=0)
    WEEK_OF_YEAR: int = Field(..., ge=1, le=53)
    QUARTER: int = Field(..., ge=1, le=4)
    DIST_X_RUSH: float = Field(..., ge=0.0)

    # Binary features
    IS_RUSH_HOUR: int = Field(..., ge=0, le=1)
    IS_WEEKEND: int = Field(..., ge=0, le=1)
    IS_LATE_NIGHT: int = Field(..., ge=0, le=1)
    IS_AIRPORT_PU: int = Field(..., ge=0, le=1)
    IS_AIRPORT_DO: int = Field(..., ge=0, le=1)
    IS_AIRPORT_TRIP: int = Field(..., ge=0, le=1)
    SAME_BOROUGH: int = Field(..., ge=0, le=1)
    PU_IS_MANHATTAN: int = Field(..., ge=0, le=1)
    DO_IS_MANHATTAN: int = Field(..., ge=0, le=1)
    RUSH_AIRPORT: int = Field(..., ge=0, le=1)
    LATE_NIGHT_WEEKEND: int = Field(..., ge=0, le=1)

    # Categorical features
    SERVICE_TYPE: str
    PU_BOROUGH: str
    DO_BOROUGH: str
    STORE_AND_FWD_FLAG: str
    VENDOR_ID: int
    PU_LOCATION_ID: int
    DO_LOCATION_ID: int
    RATE_CODE_ID: int
    PAYMENT_TYPE: int
    TRIP_TYPE: int


@app.get("/")
def root():
    return {
        "message": "NYC Taxi Fare Prediction API is running.",
        "model": "LightGBM Regressor",
        "target": "FARE_AMOUNT"
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict_trip_fare(trip: TripInput):
    try:
        input_data = trip.model_dump()
        predicted_fare = predict_fare(input_data, model=model)

        return {
            "estimated_fare_amount": round(predicted_fare, 2),
            "currency": "USD",
            "model": "LightGBM Regressor"
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )