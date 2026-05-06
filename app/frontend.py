import requests
import pandas as pd
import streamlit as st


API_URL = "http://127.0.0.1:8000/predict"

TAXI_ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

# Mapas de catálogos (ver src/data/sql/01_create_int_trips_enriched.sql).
# El frontend muestra el texto descriptivo, pero el payload a la API sigue
# enviando los IDs originales — la lógica del modelo no cambia.
VENDOR_LABELS = {
    1: "Creative Mobile Technologies, LLC",
    2: "Curb Mobility, LLC",
    3: "Unknown",
    4: "Unknown",
    5: "Unknown",
    6: "Myle Technologies Inc",
    7: "Helix",
}

PAYMENT_TYPE_LABELS = {
    0: "Flex Fare trip",
    1: "Credit card",
    2: "Cash",
    3: "No charge",
    4: "Dispute",
    5: "Unknown",
    6: "Voided trip",
}

RATE_CODE_LABELS = {
    1: "Standard rate",
    2: "JFK",
    3: "Newark",
    4: "Nassau or Westchester",
    5: "Negotiated fare",
    6: "Group ride",
    99: "Null/unknown",
}

TRIP_TYPE_LABELS = {
    1: "Street-hail",
    2: "Dispatch",
}

STORE_FWD_LABELS = {"N": "No", "Y": "Yes"}


@st.cache_data(show_spinner=False)
def load_taxi_zones() -> dict:
    """Lookup id -> 'ID - Zone (Borough)'. Cacheado para no re-fetchear."""
    try:
        df = pd.read_csv(TAXI_ZONES_URL)
        return {
            int(row.LocationID): f"{int(row.LocationID)} - {row.Zone} ({row.Borough})"
            for row in df.itertuples(index=False)
        }
    except Exception:
        return {i: f"Zone {i}" for i in range(1, 266)}


def fmt(mapping: dict):
    return lambda key: f"{key} - {mapping.get(key, 'Unknown')}"


st.set_page_config(
    page_title="NYC Taxi Fare Predictor",
    page_icon="🚕",
    layout="wide"
)

st.title("NYC Taxi Fare Predictor")
st.write(
    "This app predicts the base fare amount for a NYC taxi trip using a trained LightGBM model."
)

st.info(
    "Make sure the FastAPI server is running with: "
    "`uvicorn src.api.main:app --reload`"
)

zone_labels = load_taxi_zones()
zone_options = sorted(zone_labels.keys())
DAY_LABELS = {
    0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday",
    4: "Thursday", 5: "Friday", 6: "Saturday",
}

with st.form("fare_prediction_form"):
    st.subheader("Trip Information")

    col1, col2 = st.columns(2)

    with col1:
        service_type = st.selectbox(
            "Service Type",
            options=["yellow", "green"],
            index=0
        )

        vendor_id = st.selectbox(
            "Vendor",
            options=list(VENDOR_LABELS.keys()),
            index=1,
            format_func=fmt(VENDOR_LABELS)
        )

        pickup_hour = st.slider(
            "Pickup Hour",
            min_value=0,
            max_value=23,
            value=14
        )

        day_of_week = st.selectbox(
            "Day of Week",
            options=list(range(0, 7)),
            index=2,
            format_func=lambda d: f"{d} - {DAY_LABELS[d]}",
            help="Snowflake encoding: 0=Sunday ... 6=Saturday"
        )

        month = st.selectbox(
            "Month",
            options=list(range(1, 13)),
            index=4
        )

        year = st.selectbox(
            "Year",
            options=list(range(2020, 2026)),
            index=5
        )

        passenger_count = st.slider(
            "Passenger Count",
            min_value=1,
            max_value=6,
            value=1
        )

        trip_distance = st.number_input(
            "Trip Distance (miles)",
            min_value=0.1,
            max_value=150.0,
            value=3.2,
            step=0.1
        )

    with col2:
        pu_location_id = st.selectbox(
            "Pickup Location",
            options=zone_options,
            index=zone_options.index(237) if 237 in zone_options else 0,
            format_func=lambda i: zone_labels.get(i, f"Zone {i}")
        )

        do_location_id = st.selectbox(
            "Dropoff Location",
            options=zone_options,
            index=zone_options.index(161) if 161 in zone_options else 0,
            format_func=lambda i: zone_labels.get(i, f"Zone {i}")
        )

        pu_borough = st.selectbox(
            "Pickup Borough",
            options=["Manhattan", "Queens", "Brooklyn", "Bronx", "Staten Island", "EWR", "Unknown"],
            index=0
        )

        do_borough = st.selectbox(
            "Dropoff Borough",
            options=["Manhattan", "Queens", "Brooklyn", "Bronx", "Staten Island", "EWR", "Unknown"],
            index=0
        )

        rate_code_id = st.selectbox(
            "Rate Code",
            options=list(RATE_CODE_LABELS.keys()),
            index=0,
            format_func=fmt(RATE_CODE_LABELS)
        )

        payment_type = st.selectbox(
            "Payment Type",
            options=list(PAYMENT_TYPE_LABELS.keys()),
            index=1,
            format_func=fmt(PAYMENT_TYPE_LABELS)
        )

        trip_type = st.selectbox(
            "Trip Type",
            options=list(TRIP_TYPE_LABELS.keys()),
            index=0,
            format_func=fmt(TRIP_TYPE_LABELS)
        )

        store_and_fwd_flag = st.selectbox(
            "Store and Forward Flag",
            options=["N", "Y"],
            index=0,
            format_func=lambda v: f"{v} - {STORE_FWD_LABELS[v]}"
        )

    st.subheader("Derived Features")

    is_rush_hour = int(
        (7 <= pickup_hour <= 9) or
        (16 <= pickup_hour <= 19)
    )

    is_weekend = int(day_of_week in [0, 6])

    is_late_night = int(
        pickup_hour >= 22 or
        pickup_hour <= 5
    )

    airport_location_ids = {1, 132, 138}

    is_airport_pu = int(pu_location_id in airport_location_ids)
    is_airport_do = int(do_location_id in airport_location_ids)
    is_airport_trip = int(is_airport_pu == 1 or is_airport_do == 1)

    same_borough = int(pu_borough == do_borough)

    pu_is_manhattan = int(pu_borough == "Manhattan")
    do_is_manhattan = int(do_borough == "Manhattan")

    rush_airport = int(is_rush_hour == 1 and is_airport_trip == 1)
    late_night_weekend = int(is_late_night == 1 and is_weekend == 1)

    dist_x_rush = float(trip_distance * is_rush_hour)

    # Approximate time period encoding used by the model.
    # This should match the project feature engineering logic as closely as possible.
    if 5 <= pickup_hour <= 11:
        time_of_day = 1
    elif 12 <= pickup_hour <= 16:
        time_of_day = 2
    elif 17 <= pickup_hour <= 21:
        time_of_day = 3
    else:
        time_of_day = 0

    quarter = ((month - 1) // 3) + 1

    # Simple approximation for app input.
    # The model was trained using WEEK_OF_YEAR from pickup datetime.
    week_of_year = min(max((month - 1) * 4 + 1, 1), 53)

    with st.expander("Show derived feature values"):
        st.json({
            "IS_RUSH_HOUR": is_rush_hour,
            "IS_WEEKEND": is_weekend,
            "IS_LATE_NIGHT": is_late_night,
            "IS_AIRPORT_PU": is_airport_pu,
            "IS_AIRPORT_DO": is_airport_do,
            "IS_AIRPORT_TRIP": is_airport_trip,
            "SAME_BOROUGH": same_borough,
            "PU_IS_MANHATTAN": pu_is_manhattan,
            "DO_IS_MANHATTAN": do_is_manhattan,
            "RUSH_AIRPORT": rush_airport,
            "LATE_NIGHT_WEEKEND": late_night_weekend,
            "DIST_X_RUSH": dist_x_rush,
            "TIME_OF_DAY": time_of_day,
            "WEEK_OF_YEAR": week_of_year,
            "QUARTER": quarter
        })

    submitted = st.form_submit_button("Predict Fare")
# Initialize session state for repeated predictions
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "last_payload" not in st.session_state:
    st.session_state.last_payload = None

if "prediction_count" not in st.session_state:
    st.session_state.prediction_count = 0

if submitted:
    payload = {
        # Numeric features
        "PICKUP_HOUR": pickup_hour,
        "DAY_OF_WEEK": day_of_week,
        "MONTH": month,
        "YEAR": year,
        "PASSENGER_COUNT": passenger_count,
        "TRIP_DISTANCE": trip_distance,
        "TIME_OF_DAY": time_of_day,
        "WEEK_OF_YEAR": week_of_year,
        "QUARTER": quarter,
        "DIST_X_RUSH": dist_x_rush,

        # Binary features
        "IS_RUSH_HOUR": is_rush_hour,
        "IS_WEEKEND": is_weekend,
        "IS_LATE_NIGHT": is_late_night,
        "IS_AIRPORT_PU": is_airport_pu,
        "IS_AIRPORT_DO": is_airport_do,
        "IS_AIRPORT_TRIP": is_airport_trip,
        "SAME_BOROUGH": same_borough,
        "PU_IS_MANHATTAN": pu_is_manhattan,
        "DO_IS_MANHATTAN": do_is_manhattan,
        "RUSH_AIRPORT": rush_airport,
        "LATE_NIGHT_WEEKEND": late_night_weekend,

        # Categorical features
        "SERVICE_TYPE": service_type,
        "PU_BOROUGH": pu_borough,
        "DO_BOROUGH": do_borough,
        "STORE_AND_FWD_FLAG": store_and_fwd_flag,
        "VENDOR_ID": vendor_id,
        "PU_LOCATION_ID": int(pu_location_id),
        "DO_LOCATION_ID": int(do_location_id),
        "RATE_CODE_ID": rate_code_id,
        "PAYMENT_TYPE": payment_type,
        "TRIP_TYPE": trip_type
    }

    try:
        with st.spinner("Generating prediction..."):
            response = requests.post(
                API_URL,
                json=payload,
                timeout=20
            )

        if response.status_code == 200:
            result = response.json()

            st.session_state.last_prediction = result
            st.session_state.last_payload = payload
            st.session_state.prediction_count += 1

        else:
            st.error("API returned an error.")
            try:
                st.json(response.json())
            except Exception:
                st.write(response.text)

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the API. "
            "Make sure FastAPI is running on http://127.0.0.1:8000."
        )

    except requests.exceptions.Timeout:
        st.error("The API request timed out. Try again or restart the FastAPI server.")

    except Exception as exc:
        st.error(f"Unexpected error: {exc}")


# Always display the latest prediction stored in session state
if st.session_state.last_prediction is not None:
    estimated_fare = st.session_state.last_prediction["estimated_fare_amount"]

    st.success(
        f"Estimated Fare Amount: ${estimated_fare:.2f}"
    )

    st.caption(
        f"Prediction #{st.session_state.prediction_count}"
    )

    with st.expander("Latest API response"):
        st.json(st.session_state.last_prediction)

    with st.expander("Latest payload sent to API"):
        st.json(st.session_state.last_payload)