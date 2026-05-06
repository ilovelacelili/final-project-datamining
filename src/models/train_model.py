"""
Entrenamiento productivo del modelo final (LightGBM).

El notebook 04 evaluó AdaBoost, GradientBoosting, XGBoost, LightGBM y CatBoost
(además de Bagging/Pasting/Voting). El ganador por RMSE de validación fue
LightGBM Regressor, así que este script de producción re-entrena ese modelo con
los mejores hiperparámetros encontrados en GridSearchCV, lo evalúa contra
validación (2024) y test (2025), y guarda artefactos en data/processed/.

Para reproducir la comparación de modelos completa, ver notebook 04.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import snowflake.connector
from dotenv import load_dotenv
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# Permitir `python src/models/train_model.py` desde la raíz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_features import (
    TARGET,
    get_feature_pipeline,
    split_xy,
)

load_dotenv()

ARTIFACT_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = ARTIFACT_DIR / "best_fare_model_lightgbm_pipeline.pkl"
METADATA_PATH = ARTIFACT_DIR / "best_fare_model_lightgbm_metadata.json"
METRICS_CSV_PATH = ARTIFACT_DIR / "final_test_metrics.csv"

# Sampling para mantener el entrenamiento out-of-core friendly.
TRAIN_SAMPLE_PCT = 0.1
VAL_SAMPLE_PCT = 1
TEST_SAMPLE_PCT = 1

# Hiperparámetros ganadores del GridSearchCV en notebook 04.
LIGHTGBM_BEST_PARAMS = {
    "objective": "regression",
    "n_estimators": 200,
    "learning_rate": 0.1,
    "max_depth": 5,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "random_state": 42,
    "n_jobs": 2,
    "verbose": -1,
}


def get_conn():
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
        role=os.environ.get("SNOWFLAKE_ROLE"),
    )


def query(sql: str) -> pd.DataFrame:
    conn = get_conn()
    try:
        return pd.read_sql(sql, conn)
    finally:
        conn.close()


def fetch_split(table: str, sample_pct: float) -> pd.DataFrame:
    sql = f"SELECT * FROM ANALYTICS.{table} TABLESAMPLE ({sample_pct})"
    print(f"[load] {table} TABLESAMPLE({sample_pct})")
    df = query(sql)
    print(f"[load] {table}: {len(df):,} filas, {df.shape[1]} columnas")
    return df


def compute_metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = fetch_split("TRAIN_FE", TRAIN_SAMPLE_PCT)
    val_df = fetch_split("VAL_FE", VAL_SAMPLE_PCT)

    X_train, y_train = split_xy(train_df)
    X_val, y_val = split_xy(val_df)
    X_val = X_val[X_train.columns]

    preprocessor, feature_groups = get_feature_pipeline(X_train)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LGBMRegressor(**LIGHTGBM_BEST_PARAMS)),
    ])

    print("[train] fitting LightGBM pipeline...")
    start = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"[train] done in {train_time:.2f}s")

    train_metrics = compute_metrics(y_train, pipeline.predict(X_train))
    val_metrics = compute_metrics(y_val, pipeline.predict(X_val))
    print(f"[val]   RMSE={val_metrics['rmse']:.4f}  MAE={val_metrics['mae']:.4f}  R2={val_metrics['r2']:.4f}")

    test_df = fetch_split("TEST_FE", TEST_SAMPLE_PCT)
    X_test, y_test = split_xy(test_df)
    X_test = X_test[X_train.columns]
    test_metrics = compute_metrics(y_test, pipeline.predict(X_test))
    print(f"[test]  RMSE={test_metrics['rmse']:.4f}  MAE={test_metrics['mae']:.4f}  R2={test_metrics['r2']:.4f}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"[save] model -> {MODEL_PATH}")

    metadata = {
        "project": "NYC Taxi Trips Fare Prediction",
        "target": TARGET,
        "selected_model": "LightGBM Regressor",
        "selection_metric": "validation_rmse",
        "model_artifact": MODEL_PATH.name,
        "artifact_type": "sklearn_pipeline_with_preprocessor_and_lightgbm",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_sources": {
            "train_table": "ANALYTICS.TRAIN_FE",
            "validation_table": "ANALYTICS.VAL_FE",
            "test_table": "ANALYTICS.TEST_FE",
        },
        "data_splits": {
            "train": "2015-2023",
            "validation": "2024",
            "test": "2025",
        },
        "sampling": {
            "train_sample_pct": TRAIN_SAMPLE_PCT,
            "val_sample_pct": VAL_SAMPLE_PCT,
            "test_sample_pct": TEST_SAMPLE_PCT,
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
        },
        "validation_metrics": {
            "train_rmse": train_metrics["rmse"],
            "val_rmse": val_metrics["rmse"],
            "train_mae": train_metrics["mae"],
            "val_mae": val_metrics["mae"],
            "train_r2": train_metrics["r2"],
            "val_r2": val_metrics["r2"],
            "train_time_sec": train_time,
        },
        "test_metrics": {
            "test_sample_percent": float(TEST_SAMPLE_PCT),
            "test_rows": int(len(test_df)),
            "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "test_r2": test_metrics["r2"],
        },
        "lightgbm_best_params": LIGHTGBM_BEST_PARAMS,
        "feature_groups": feature_groups,
        "library_versions": {
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
            "lightgbm": __import__("lightgbm").__version__,
        },
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[save] metadata -> {METADATA_PATH}")

    metrics_row = {
        "model_name": "LightGBM Regressor",
        "train_rmse": train_metrics["rmse"],
        "val_rmse": val_metrics["rmse"],
        "test_rmse": test_metrics["rmse"],
        "train_mae": train_metrics["mae"],
        "val_mae": val_metrics["mae"],
        "test_mae": test_metrics["mae"],
        "train_r2": train_metrics["r2"],
        "val_r2": val_metrics["r2"],
        "test_r2": test_metrics["r2"],
        "train_time_sec": train_time,
    }
    pd.DataFrame([metrics_row]).to_csv(METRICS_CSV_PATH, index=False)
    print(f"[save] metrics csv -> {METRICS_CSV_PATH}")


if __name__ == "__main__":
    main()
