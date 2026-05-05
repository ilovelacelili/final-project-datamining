import os
import pandas as pd
import numpy as np
import time
from dotenv import load_dotenv
import snowflake.connector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor

load_dotenv()

def evaluate_model(name, model, X_train, y_train, X_val, y_val):
    """
    Trains a model inside a preprocessing pipeline and evaluates it
    on both training and validation data.

    Parameters
    ----------
    name : str
        Model name.
    model : sklearn-compatible estimator
        Regression model to train.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    X_val : pd.DataFrame
        Validation features.
    y_val : pd.Series
        Validation target.

    Returns
    -------
    dict
        Dictionary containing model metrics and the trained pipeline.
    """

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    start_time = time.time()

    pipeline.fit(X_train, y_train)

    train_time = time.time() - start_time

    train_preds = pipeline.predict(X_train)
    val_preds = pipeline.predict(X_val)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

    train_mae = mean_absolute_error(y_train, train_preds)
    val_mae = mean_absolute_error(y_val, val_preds)

    train_r2 = r2_score(y_train, train_preds)
    val_r2 = r2_score(y_val, val_preds)

    result = {
        "model_name": name,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "train_mae": train_mae,
        "val_mae": val_mae,
        "train_r2": train_r2,
        "val_r2": val_r2,
        "train_time_sec": train_time,
        "pipeline": pipeline
    }

    return result

def get_conn():
    return snowflake.connector.connect(
        account=os.environ['SNOWFLAKE_ACCOUNT'],
        user=os.environ['SNOWFLAKE_USER'],
        password=os.environ['SNOWFLAKE_PASSWORD'],
        database=os.environ['SNOWFLAKE_DATABASE'],
        warehouse=os.environ['SNOWFLAKE_WAREHOUSE'],
        schema=os.environ['SNOWFLAKE_SCHEMA'],
        role=os.environ.get('SNOWFLAKE_ROLE'),
    )

def query(sql: str) -> pd.DataFrame:
    conn = get_conn()
    try:
        return pd.read_sql(sql, conn)
    finally:
        conn.close()

if __name__== "__main__":
    train_df = query("""
    SELECT *
    FROM ANALYTICS.TRAIN_FE
    TABLESAMPLE (0.1)
""")

    val_df = query("""
        SELECT *
        FROM ANALYTICS.VAL_FE
        TABLESAMPLE (1)
    """)

    TARGET = 'FARE_AMOUNT'
    DROP_COLS = [
        # target
        'FARE_AMOUNT',

        # leakage
        'DROPOFF_DATETIME',
        'DROPOFF_DATE',
        'DROPOFF_HOUR',
        'TRIP_DURATION_MIN',
        'AVG_SPEED_MPH',
        'TIP_AMOUNT',
        'TIP_PCT',
        'EXTRA',
        'MTA_TAX',
        'TOLLS_AMOUNT',
        'IMPROVEMENT_SURCHARGE',
        'CONGESTION_SURCHARGE',
        'AIRPORT_FEE',
        'EHAIL_FEE',
        'TOTAL_AMOUNT',

        # metadata / linaje
        'RUN_ID',
        'SOURCE_YEAR',
        'SOURCE_MONTH',
        'SOURCE_PATH',
        'INGESTED_AT_UTC',

        # datetime crudo, mejor usar columnas derivadas
        'PICKUP_DATETIME',
        'PICKUP_DATE'
    ]

    X_train = train_df.drop(columns=[c for c in DROP_COLS if c in train_df.columns])
    y_train = train_df[TARGET]

    X_val = val_df.drop(columns=[c for c in DROP_COLS if c in val_df.columns])
    y_val = val_df[TARGET]



    # Asegurar que X_val tenga exactamente las mismas columnas que X_train
    X_val = X_val[X_train.columns]

    # Columnas textuales redundantes que ya tienen una versión codificada
    REDUNDANT_TEXT_COLS = [
        'VENDOR_NAME',
        'PU_ZONE',
        'DO_ZONE',
        'RATE_CODE_DESC',
        'PAYMENT_TYPE_DESC'
    ]

    X_train = X_train.drop(columns=[c for c in REDUNDANT_TEXT_COLS if c in X_train.columns])
    X_val = X_val.drop(columns=[c for c in REDUNDANT_TEXT_COLS if c in X_val.columns])

    # 1. Detectar columnas categóricas por tipo de dato
    categorical_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # 2. Detectar columnas numéricas
    numeric_candidate_features = X_train.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    # 3. Columnas numéricas que realmente son categóricas por ser IDs, códigos o tipos
    categorical_name_patterns = [
        "ID",
        "TYPE",
        "CODE",
        "FLAG",
        "SERVICE"
    ]

    numeric_but_categorical = [
        col for col in numeric_candidate_features
        if any(pattern in col.upper() for pattern in categorical_name_patterns)
    ]

    # 4. Detectar columnas binarias 0/1
    binary_features = []

    for col in numeric_candidate_features:
        if col not in numeric_but_categorical:
            unique_values = set(X_train[col].dropna().unique())
            if unique_values.issubset({0, 1}):
                binary_features.append(col)

    # 5. Las numéricas finales son las numéricas que no son categóricas ni binarias
    numeric_features = [
        col for col in numeric_candidate_features
        if col not in numeric_but_categorical
        and col not in binary_features
    ]

    # 6. Agregar las numéricas categóricas a las categóricas finales
    categorical_features = categorical_features + numeric_but_categorical

    # 7. Quitar duplicados por seguridad
    numeric_features = list(dict.fromkeys(numeric_features))
    binary_features = list(dict.fromkeys(binary_features))
    categorical_features = list(dict.fromkeys(categorical_features))





    # Pipeline para variables numéricas continuas
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Pipeline para variables binarias 0/1
    binary_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    # Pipeline para variables categóricas
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    # Preprocesador general
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("bin", binary_pipeline, binary_features),
            ("cat", categorical_pipeline, categorical_features)
        ],
        remainder="drop"
    )

    best_lightgbm_model = LGBMRegressor(
        objective="regression",
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=42,
        n_jobs=2,
        verbose=-1
    )

    lightgbm_final_result = evaluate_model(
        name="LightGBM Regressor",
        model=best_lightgbm_model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val
    )

    print(f"LightGBM Regressor validation RMSE: {lightgbm_final_result['val_rmse']:.4f}")
    print(f"LightGBM Regressor validation MAE: {lightgbm_final_result['val_mae']:.4f}")
    print(f"LightGBM Regressor validation R2: {lightgbm_final_result['val_r2']:.4f}")
    print(f"Training time: {lightgbm_final_result['train_time_sec']:.2f} seconds")
    print("-" * 50)