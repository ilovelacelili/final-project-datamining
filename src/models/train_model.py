"""
Entrenamiento productivo del modelo final (LightGBM).

El notebook 04 evaluó AdaBoost, GradientBoosting, XGBoost, LightGBM y CatBoost
(además de Bagging/Pasting/Voting). El ganador por RMSE de validación fue
LightGBM Regressor, así que este script de producción re-entrena ese modelo con
los mejores hiperparámetros encontrados en GridSearchCV, lo evalúa contra
validación (2024) y test (2025), y guarda artefactos en data/processed/.

Sampling:
- TRAIN: muestreo estratificado por (SERVICE_TYPE, YEAR) — corrige el sesgo
  Yellow/Green y pre-COVID detectado en el EDA. Misma lógica que el notebook 04.
- VAL/TEST: TABLESAMPLE uniforme — la evaluación debe reflejar la distribución
  real del año futuro.

Caché: cada split se persiste en data/interim/ como parquet. Re-ejecuciones
posteriores leen del parquet en vez de volver a Snowflake (la query
estratificada de train tarda ~30 min).
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
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
from src.utils.config import query

ARTIFACT_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
MODEL_PATH = ARTIFACT_DIR / "best_fare_model_lightgbm_pipeline.pkl"
METADATA_PATH = ARTIFACT_DIR / "best_fare_model_lightgbm_metadata.json"
METRICS_CSV_PATH = ARTIFACT_DIR / "final_test_metrics.csv"

# Sampling.
TRAIN_PER_STRATUM = 35_000  # 18 estratos x 35k ≈ 630k filas (~0.1% balanceado)
VAL_SAMPLE_PCT = 1
TEST_SAMPLE_PCT = 1

# Si está en True, ignora los parquets cacheados y vuelve a consultar Snowflake.
FORCE_REFRESH = False

TRAIN_PARQUET = INTERIM_DIR / f"train_fe_stratified_{TRAIN_PER_STRATUM}.parquet"
VAL_PARQUET = INTERIM_DIR / f"val_fe_sample_{VAL_SAMPLE_PCT}pct.parquet"
TEST_PARQUET = INTERIM_DIR / f"test_fe_sample_{TEST_SAMPLE_PCT}pct.parquet"

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


def _load_or_query(parquet_path: Path, snowflake_sql: str, label: str) -> pd.DataFrame:
    """Carga desde parquet local si existe; si no, va a Snowflake y guarda el resultado."""
    if parquet_path.exists() and not FORCE_REFRESH:
        print(f"[cache] {label}: leyendo {parquet_path}")
        return pd.read_parquet(parquet_path)

    print(f"[snowflake] {label}: descargando...")
    df = query(snowflake_sql)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    print(f"[cache] {label}: guardado -> {parquet_path}")
    return df


def fetch_train_stratified() -> pd.DataFrame:
    """
    Muestreo estratificado de TRAIN_FE por (SERVICE_TYPE, YEAR).

    SELECT puro con QUALIFY ROW_NUMBER — no crea ni modifica tablas.
    RANDOM(42) fija la semilla para reproducibilidad.
    """
    sql = f"""
        SELECT *
        FROM ANALYTICS.TRAIN_FE
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY SERVICE_TYPE, YEAR
            ORDER BY RANDOM(42)
        ) <= {TRAIN_PER_STRATUM}
    """
    df = _load_or_query(TRAIN_PARQUET, sql, f"TRAIN_FE stratified (cap={TRAIN_PER_STRATUM:,}/estrato)")
    if {"SERVICE_TYPE", "YEAR"}.issubset(df.columns):
        counts = df.groupby(["SERVICE_TYPE", "YEAR"]).size().reset_index(name="rows")
        print("[load] distribución por estrato:")
        print(counts.to_string(index=False))
    return df


def fetch_val() -> pd.DataFrame:
    sql = f"SELECT * FROM ANALYTICS.VAL_FE TABLESAMPLE ({VAL_SAMPLE_PCT})"
    return _load_or_query(VAL_PARQUET, sql, f"VAL_FE TABLESAMPLE({VAL_SAMPLE_PCT})")


def fetch_test() -> pd.DataFrame:
    sql = f"SELECT * FROM ANALYTICS.TEST_FE TABLESAMPLE ({TEST_SAMPLE_PCT})"
    return _load_or_query(TEST_PARQUET, sql, f"TEST_FE TABLESAMPLE({TEST_SAMPLE_PCT})")


def compute_metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = fetch_train_stratified()
    val_df = fetch_val()
    print(f"train_df: {len(train_df):,} filas")
    print(f"val_df:   {len(val_df):,} filas")

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

    test_df = fetch_test()
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
            "train_strategy": "stratified_by_service_type_and_year",
            "train_per_stratum_cap": TRAIN_PER_STRATUM,
            "train_random_seed": 42,
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
