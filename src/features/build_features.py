"""
Pipeline de features de producción.

Centraliza la lógica que en notebook 04 vivía dispersa:
- columnas a eliminar (target, leakage, metadata, datetime crudo)
- columnas textuales redundantes (ya hay versión codificada)
- detección automática de grupos de features (numéricas, binarias, categóricas)
- construcción del ColumnTransformer

Importado por src/models/train_model.py y, vía el .pkl, por src/api.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "FARE_AMOUNT"

# Columnas que NO deben entrar como features:
# - target
# - leakage (solo conocidas al cierre del viaje)
# - metadata / linaje
# - datetime crudo (ya hay derivadas como PICKUP_HOUR, MONTH, etc.)
DROP_COLS = [
    # target
    "FARE_AMOUNT",
    # leakage
    "DROPOFF_DATETIME", "DROPOFF_DATE", "DROPOFF_HOUR",
    "TRIP_DURATION_MIN", "AVG_SPEED_MPH",
    "TIP_AMOUNT", "TIP_PCT",
    "EXTRA", "MTA_TAX", "TOLLS_AMOUNT",
    "IMPROVEMENT_SURCHARGE", "CONGESTION_SURCHARGE",
    "AIRPORT_FEE", "EHAIL_FEE", "TOTAL_AMOUNT",
    # metadata / linaje
    "RUN_ID", "SOURCE_YEAR", "SOURCE_MONTH",
    "SOURCE_PATH", "INGESTED_AT_UTC",
    # datetime crudo
    "PICKUP_DATETIME", "PICKUP_DATE",
]

# Columnas textuales redundantes (ya existe versión codificada como ID)
REDUNDANT_TEXT_COLS = [
    "VENDOR_NAME",
    "PU_ZONE", "DO_ZONE",
    "RATE_CODE_DESC",
    "PAYMENT_TYPE_DESC",
]

# Patrones para identificar columnas numéricas que en realidad son categóricas
CATEGORICAL_NAME_PATTERNS = ["ID", "TYPE", "CODE", "FLAG", "SERVICE"]


def split_xy(df: pd.DataFrame, target: str = TARGET) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa features y target, eliminando columnas de leakage/metadata/redundantes."""
    y = df[target]
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    X = X.drop(columns=[c for c in REDUNDANT_TEXT_COLS if c in X.columns])
    return X, y


def detect_feature_groups(X: pd.DataFrame) -> dict:
    """
    Detecta tres grupos de features:
    - numeric_features: continuas (escalado + imputación mediana)
    - binary_features: 0/1 (solo imputación)
    - categorical_features: object/category + numéricas-pero-categóricas (OHE)
    """
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_candidates = X.select_dtypes(include=[np.number]).columns.tolist()

    numeric_but_categorical = [
        col for col in numeric_candidates
        if any(p in col.upper() for p in CATEGORICAL_NAME_PATTERNS)
    ]

    binary_features = []
    for col in numeric_candidates:
        if col in numeric_but_categorical:
            continue
        unique_values = set(X[col].dropna().unique())
        if unique_values and unique_values.issubset({0, 1}):
            binary_features.append(col)

    numeric_features = [
        col for col in numeric_candidates
        if col not in numeric_but_categorical and col not in binary_features
    ]

    categorical_features = list(dict.fromkeys(categorical_features + numeric_but_categorical))
    numeric_features = list(dict.fromkeys(numeric_features))
    binary_features = list(dict.fromkeys(binary_features))

    return {
        "numeric_features": numeric_features,
        "binary_features": binary_features,
        "categorical_features": categorical_features,
    }


def build_preprocessor(feature_groups: dict) -> ColumnTransformer:
    """ColumnTransformer con pipelines separados para num, bin y cat."""
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    binary_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, feature_groups["numeric_features"]),
            ("bin", binary_pipeline, feature_groups["binary_features"]),
            ("cat", categorical_pipeline, feature_groups["categorical_features"]),
        ],
        remainder="drop",
    )


def get_feature_pipeline(X: pd.DataFrame) -> Tuple[ColumnTransformer, dict]:
    """Atajo: detecta grupos y devuelve (preprocessor, feature_groups)."""
    feature_groups = detect_feature_groups(X)
    preprocessor = build_preprocessor(feature_groups)
    return preprocessor, feature_groups
