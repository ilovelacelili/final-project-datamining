from pathlib import Path
import json
import joblib
import pandas as pd


# Project root:
# src/models/predict_model.py -> parents[2] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "best_fare_model_lightgbm_pipeline.pkl"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "best_fare_model_lightgbm_metadata.json"


def load_model(model_path: Path = MODEL_PATH):
    """
    Load the trained sklearn pipeline.

    The saved artifact includes:
    - preprocessing pipeline
    - trained LightGBM model
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return joblib.load(model_path)


def load_metadata(metadata_path: Path = METADATA_PATH) -> dict:
    """
    Load model metadata exported from notebook 04.
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_expected_features() -> list:
    """
    Return the full list of features expected by the trained pipeline.
    """
    metadata = load_metadata()

    feature_groups = metadata["feature_groups"]

    expected_features = (
        feature_groups["numeric_features"]
        + feature_groups["binary_features"]
        + feature_groups["categorical_features"]
    )

    return expected_features


def validate_input(input_data: dict, expected_features: list) -> dict:
    """
    Validate that all required model features are present.
    Extra fields are ignored.
    """
    missing_features = [
        feature for feature in expected_features
        if feature not in input_data
    ]

    if missing_features:
        raise ValueError(
            "Missing required features: "
            + ", ".join(missing_features)
        )

    validated_data = {
        feature: input_data[feature]
        for feature in expected_features
    }

    return validated_data


def predict_fare(input_data: dict, model=None) -> float:
    """
    Predict fare amount for a single trip.

    Parameters
    ----------
    input_data : dict
        Dictionary containing the same feature columns used during training.
    model : sklearn Pipeline, optional
        Preloaded model. If None, the model is loaded from disk.

    Returns
    -------
    float
        Predicted fare amount.
    """
    if model is None:
        model = load_model()

    expected_features = get_expected_features()
    validated_data = validate_input(input_data, expected_features)

    input_df = pd.DataFrame([validated_data], columns=expected_features)

    prediction = model.predict(input_df)[0]

    return float(prediction)


if __name__ == "__main__":
    metadata = load_metadata()
    expected_features = get_expected_features()

    print("Model loaded from:")
    print(MODEL_PATH)

    print("\nSelected model:")
    print(metadata["selected_model"])

    print("\nExpected features:")
    for feature in expected_features:
        print("-", feature)