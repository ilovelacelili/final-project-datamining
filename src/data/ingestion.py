

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import pandas as pd
import pyarrow.parquet as pq
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas


try:
    from src.utils.config import get_snowflake_credentials
except ImportError:
    def get_snowflake_credentials() -> Dict[str, Optional[str]]:
        """
        Fallback si no existe src.utils.config.

        Si tu proyecto ya tiene src.utils.config.get_snowflake_credentials(),
        se usará automáticamente esa función.
        """
        return {
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "user": os.getenv("SNOWFLAKE_USER"),
            "password": os.getenv("SNOWFLAKE_PASSWORD"),
            "database": os.getenv("SNOWFLAKE_DATABASE"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
            "role": os.getenv("SNOWFLAKE_ROLE", "SYSADMIN"),
        }


# =============================================================================
# Configuración
# =============================================================================

def clean_identifier(value: str, label: str) -> str:
    """
    Valida identificadores SQL para evitar SQL dinámico inseguro.

    Ejemplos válidos:
        RAW
        ANALYTICS
        NYC_TAXI_DB
    """
    if value is None:
        raise ValueError(f"Falta identificador para {label}")

    value = value.strip()

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Identificador inválido para {label}: {value!r}")

    return value.upper()


def env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "si", "sí"}


def get_credential_value(creds: Any, key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Lee credenciales desde dict o desde objeto con atributos.

    Soporta:
        creds["account"]
        creds["SNOWFLAKE_ACCOUNT"]
        creds.account
        creds.SNOWFLAKE_ACCOUNT
    """
    candidates = [
        key,
        key.lower(),
        key.upper(),
        f"SNOWFLAKE_{key.upper()}",
    ]

    if isinstance(creds, dict):
        for candidate in candidates:
            if candidate in creds and creds[candidate] is not None:
                return str(creds[candidate])

    for candidate in candidates:
        if hasattr(creds, candidate):
            value = getattr(creds, candidate)
            if value is not None:
                return str(value)

    return default


@dataclass(frozen=True)
class PipelineConfig:
    account: str
    user: str
    password: str
    database: str
    warehouse: str
    role: str
    raw_schema: str
    analytics_schema: str

    start_year: int
    end_year: int
    months: Tuple[int, ...]
    services: Tuple[str, ...]

    batch_size: int
    data_dir: Path
    run_id: str

    parquet_base_url: str
    taxi_zones_url: str

    filter_raw_rows: bool
    filter_enriched_rows: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingesta RAW + filtrado + enriquecimiento de NYC TLC Parquets en Snowflake."
    )

    parser.add_argument("--start-year", type=int, default=int(os.getenv("START_YEAR", "2015")))
    parser.add_argument("--end-year", type=int, default=int(os.getenv("END_YEAR", "2025")))

    parser.add_argument(
        "--months",
        default=os.getenv("MONTHS", "1,2,3,4,5,6,7,8,9,10,11,12"),
        help="Meses separados por coma. Ejemplo: 1,2,3",
    )

    parser.add_argument(
        "--services",
        default=os.getenv("SERVICES", "yellow,green"),
        help="Servicios separados por coma: yellow,green",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("BATCH_SIZE", "100000")),
        help="Tamaño de lote para lectura de Parquet e inserción.",
    )

    parser.add_argument(
        "--data-dir",
        default=os.getenv("DATA_DIR", "./data/parquet_raw"),
        help="Carpeta local temporal para descargar Parquets.",
    )

    parser.add_argument(
        "--run-id",
        default=os.getenv("RUN_ID"),
        help="Identificador de corrida. Si no se pasa, se genera automáticamente.",
    )

    parser.add_argument(
        "--filter-raw-rows",
        action="store_true",
        default=env_bool("FILTER_RAW_ROWS", "false"),
        help="Si se activa, filtra filas inválidas antes de guardar RAW.",
    )

    parser.add_argument(
        "--no-filter-enriched-rows",
        action="store_true",
        help="Si se activa, NO filtra filas inválidas en la tabla enriquecida.",
    )

    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="No ingesta Parquets RAW. Solo ejecuta enriquecimiento.",
    )

    parser.add_argument(
        "--skip-enrichment",
        action="store_true",
        help="Solo ingesta RAW. No crea catálogos ni tabla enriquecida.",
    )

    return parser.parse_args()


def load_config(args: argparse.Namespace) -> PipelineConfig:
    creds = get_snowflake_credentials()

    account = get_credential_value(creds, "account")
    user = get_credential_value(creds, "user")
    password = get_credential_value(creds, "password")
    database = get_credential_value(creds, "database")
    warehouse = get_credential_value(creds, "warehouse")
    role = get_credential_value(creds, "role", "SYSADMIN")

    missing = []
    if not account:
        missing.append("SNOWFLAKE_ACCOUNT")
    if not user:
        missing.append("SNOWFLAKE_USER")
    if not password:
        missing.append("SNOWFLAKE_PASSWORD")
    if not database:
        missing.append("SNOWFLAKE_DATABASE")
    if not warehouse:
        missing.append("SNOWFLAKE_WAREHOUSE")

    if missing:
        raise RuntimeError(f"Faltan credenciales obligatorias: {', '.join(missing)}")

    services = tuple(s.strip().lower() for s in args.services.split(",") if s.strip())
    invalid_services = [s for s in services if s not in {"yellow", "green"}]
    if invalid_services:
        raise ValueError(f"Servicios inválidos: {invalid_services}. Use yellow, green o ambos.")

    months = tuple(int(m.strip()) for m in args.months.split(",") if m.strip())
    invalid_months = [m for m in months if m < 1 or m > 12]
    if invalid_months:
        raise ValueError(f"Meses inválidos: {invalid_months}. Use valores entre 1 y 12.")

    run_id = args.run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    return PipelineConfig(
        account=account,
        user=user,
        password=password,
        database=clean_identifier(database, "SNOWFLAKE_DATABASE"),
        warehouse=clean_identifier(warehouse, "SNOWFLAKE_WAREHOUSE"),
        role=clean_identifier(role or "SYSADMIN", "SNOWFLAKE_ROLE"),
        raw_schema=clean_identifier(os.getenv("SNOWFLAKE_SCHEMA_RAW", "RAW"), "SNOWFLAKE_SCHEMA_RAW"),
        analytics_schema=clean_identifier(
            os.getenv("SNOWFLAKE_SCHEMA_ANALYTICS", "ANALYTICS"),
            "SNOWFLAKE_SCHEMA_ANALYTICS",
        ),
        start_year=args.start_year,
        end_year=args.end_year,
        months=months,
        services=services,
        batch_size=args.batch_size,
        data_dir=Path(args.data_dir),
        run_id=run_id,
        parquet_base_url=os.getenv(
            "PARQUET_BASE_URL",
            "https://d37ci6vzurychx.cloudfront.net/trip-data",
        ),
        taxi_zones_url=os.getenv(
            "TAXI_ZONES_URL",
            "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv",
        ),
        filter_raw_rows=args.filter_raw_rows,
        filter_enriched_rows=not args.no_filter_enriched_rows and env_bool(
            "FILTER_ENRICHED_ROWS",
            "true",
        ),
    )


# =============================================================================
# Snowflake: conexión y extracción por batches
# =============================================================================

def get_snowflake_connection(
    config: Optional[PipelineConfig] = None,
    schema: Optional[str] = None,
):
    """
    Establece y retorna una conexión a Snowflake.

    Si se pasa config, usa la configuración del pipeline.
    Si no se pasa config, lee desde get_snowflake_credentials().
    """
    if config is None:
        creds = get_snowflake_credentials()

        account = get_credential_value(creds, "account")
        user = get_credential_value(creds, "user")
        password = get_credential_value(creds, "password")
        database = get_credential_value(creds, "database")
        warehouse = get_credential_value(creds, "warehouse")
        role = get_credential_value(creds, "role", "SYSADMIN")

        if not all([account, user, password, database, warehouse]):
            raise ConnectionError("No se pudo construir la conexión a Snowflake. Faltan credenciales.")

        return snowflake.connector.connect(
            account=account,
            user=user,
            password=password,
            database=database,
            warehouse=warehouse,
            role=role,
            schema=schema,
        )

    return snowflake.connector.connect(
        account=config.account,
        user=config.user,
        password=config.password,
        database=config.database,
        warehouse=config.warehouse,
        role=config.role,
        schema=schema or config.raw_schema,
    )


def fetch_data_in_batches(query: str, batch_size: int = 100000) -> Iterator[pd.DataFrame]:
    """
    Extrae datos de Snowflake en batches.

    Esto evita cargar todo un dataset grande en memoria.

    Uso:
        for batch_df in fetch_data_in_batches("SELECT * FROM ANALYTICS.INT_TRIPS_ENRICHED"):
            entrenar_o_procesar(batch_df)
    """
    conn = get_snowflake_connection()
    cursor = None

    try:
        cursor = conn.cursor()
        cursor.execute(query)

        columns = [desc[0] for desc in cursor.description]

        while True:
            rows = cursor.fetchmany(batch_size)

            if not rows:
                break

            yield pd.DataFrame.from_records(rows, columns=columns)

    finally:
        if cursor is not None:
            cursor.close()
        conn.close()


def fetch_sample(query: str, sample_prob: float = 1.0) -> pd.DataFrame:
    """
    Extrae una muestra para EDA o experimentación.

    sample_prob debe estar entre 0 y 1.
    Ejemplo:
        sample_prob=0.01 equivale aproximadamente al 1%.
    """
    if sample_prob <= 0 or sample_prob > 1:
        raise ValueError("sample_prob debe estar en el rango (0, 1].")

    base_query = query.strip().rstrip(";")

    if sample_prob < 1.0:
        sample_query = f"""
        SELECT *
        FROM ({base_query}) AS q
        WHERE UNIFORM(0::FLOAT, 1::FLOAT, RANDOM()) < {sample_prob}
        """
    else:
        sample_query = base_query

    conn = get_snowflake_connection()
    cursor = None

    try:
        cursor = conn.cursor()
        cursor.execute(sample_query)

        try:
            return cursor.fetch_pandas_all()
        except Exception:
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return pd.DataFrame.from_records(rows, columns=columns)

    finally:
        if cursor is not None:
            cursor.close()
        conn.close()


def execute_sql(config: PipelineConfig, sql: str, schema: Optional[str] = None) -> None:
    conn = get_snowflake_connection(config, schema=schema)
    cursor = None

    try:
        cursor = conn.cursor()
        cursor.execute(sql)
    finally:
        if cursor is not None:
            cursor.close()
        conn.close()


def query_one(config: PipelineConfig, sql: str, schema: Optional[str] = None) -> Any:
    conn = get_snowflake_connection(config, schema=schema)
    cursor = None

    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        row = cursor.fetchone()
        return row[0] if row else None
    finally:
        if cursor is not None:
            cursor.close()
        conn.close()


# =============================================================================
# DDL Snowflake
# =============================================================================

RAW_TABLE_COLUMNS_SQL = """
    VENDORID NUMBER(38, 0),
    PICKUP_DATETIME TIMESTAMP_NTZ,
    DROPOFF_DATETIME TIMESTAMP_NTZ,
    PASSENGER_COUNT FLOAT,
    TRIP_DISTANCE FLOAT,
    RATECODEID FLOAT,
    STORE_AND_FWD_FLAG VARCHAR,
    PULOCATIONID NUMBER(38, 0),
    DOLOCATIONID NUMBER(38, 0),
    PAYMENT_TYPE FLOAT,
    FARE_AMOUNT FLOAT,
    EXTRA FLOAT,
    MTA_TAX FLOAT,
    TIP_AMOUNT FLOAT,
    TOLLS_AMOUNT FLOAT,
    IMPROVEMENT_SURCHARGE FLOAT,
    TOTAL_AMOUNT FLOAT,
    CONGESTION_SURCHARGE FLOAT,
    AIRPORT_FEE FLOAT,
    EHAIL_FEE FLOAT,
    TRIP_TYPE FLOAT,
    SERVICE_TYPE VARCHAR,
    RUN_ID VARCHAR,
    SOURCE_YEAR NUMBER(38, 0),
    SOURCE_MONTH NUMBER(38, 0),
    SOURCE_PATH VARCHAR,
    INGESTED_AT_UTC TIMESTAMP_NTZ
"""

RAW_COLUMNS = [
    "VENDORID",
    "PICKUP_DATETIME",
    "DROPOFF_DATETIME",
    "PASSENGER_COUNT",
    "TRIP_DISTANCE",
    "RATECODEID",
    "STORE_AND_FWD_FLAG",
    "PULOCATIONID",
    "DOLOCATIONID",
    "PAYMENT_TYPE",
    "FARE_AMOUNT",
    "EXTRA",
    "MTA_TAX",
    "TIP_AMOUNT",
    "TOLLS_AMOUNT",
    "IMPROVEMENT_SURCHARGE",
    "TOTAL_AMOUNT",
    "CONGESTION_SURCHARGE",
    "AIRPORT_FEE",
    "EHAIL_FEE",
    "TRIP_TYPE",
    "SERVICE_TYPE",
    "RUN_ID",
    "SOURCE_YEAR",
    "SOURCE_MONTH",
    "SOURCE_PATH",
    "INGESTED_AT_UTC",
]

NUMERIC_COLUMNS = [
    "VENDORID",
    "PASSENGER_COUNT",
    "TRIP_DISTANCE",
    "RATECODEID",
    "PULOCATIONID",
    "DOLOCATIONID",
    "PAYMENT_TYPE",
    "FARE_AMOUNT",
    "EXTRA",
    "MTA_TAX",
    "TIP_AMOUNT",
    "TOLLS_AMOUNT",
    "IMPROVEMENT_SURCHARGE",
    "TOTAL_AMOUNT",
    "CONGESTION_SURCHARGE",
    "AIRPORT_FEE",
    "EHAIL_FEE",
    "TRIP_TYPE",
    "SOURCE_YEAR",
    "SOURCE_MONTH",
]

DATETIME_COLUMNS = [
    "PICKUP_DATETIME",
    "DROPOFF_DATETIME",
    "INGESTED_AT_UTC",
]


def setup_snowflake(config: PipelineConfig) -> None:
    """
    Crea warehouse, database, schemas y tablas base si no existen.
    """
    print("Configurando Snowflake...")

    conn = snowflake.connector.connect(
        account=config.account,
        user=config.user,
        password=config.password,
        role=config.role,
    )

    cursor = None

    try:
        cursor = conn.cursor()

        cursor.execute(
            f"""
            CREATE WAREHOUSE IF NOT EXISTS {config.warehouse}
            WITH
                WAREHOUSE_SIZE = 'XSMALL'
                AUTO_SUSPEND = 60
                AUTO_RESUME = TRUE
            """
        )

        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {config.database}")
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {config.database}.{config.raw_schema}")
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {config.database}.{config.analytics_schema}")

        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {config.database}.{config.raw_schema}.TRIPS_YELLOW (
                {RAW_TABLE_COLUMNS_SQL}
            )
            """
        )

        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {config.database}.{config.raw_schema}.TRIPS_GREEN (
                {RAW_TABLE_COLUMNS_SQL}
            )
            """
        )

        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {config.database}.{config.raw_schema}.INGESTION_AUDIT (
                RUN_ID VARCHAR,
                SERVICE_TYPE VARCHAR,
                SOURCE_YEAR NUMBER(38, 0),
                SOURCE_MONTH NUMBER(38, 0),
                STATUS VARCHAR,
                ROW_COUNT NUMBER(38, 0),
                ROW_COUNT_AFTER_FILTER NUMBER(38, 0),
                ISSUES_FOUND NUMBER(38, 0),
                NULL_TIMESTAMPS NUMBER(38, 0),
                BAD_TIMESTAMPS NUMBER(38, 0),
                RANGE_VIOLATIONS NUMBER(38, 0),
                LOAD_TIME_SEC FLOAT,
                DELETED_PREVIOUS_ROWS NUMBER(38, 0),
                FILTER_RAW_ROWS BOOLEAN,
                SOURCE_PATH VARCHAR,
                ERROR VARCHAR,
                CREATED_AT_UTC TIMESTAMP_NTZ
            )
            """
        )

        print(f"Snowflake listo: {config.database}.{config.raw_schema} y {config.database}.{config.analytics_schema}")

    finally:
        if cursor is not None:
            cursor.close()
        conn.close()


# =============================================================================
# Utilidades de descarga y lectura Parquet por batches
# =============================================================================

def download_parquet(config: PipelineConfig, service: str, year: int, month: int) -> Optional[Path]:
    """
    Descarga un Parquet mensual de NYC TLC.
    Retorna None si el archivo no existe, por ejemplo meses futuros o faltantes.
    """
    filename = f"{service}_tripdata_{year}-{month:02d}.parquet"
    url = f"{config.parquet_base_url}/{filename}"

    config.data_dir.mkdir(parents=True, exist_ok=True)
    local_path = config.data_dir / filename

    print(f"Descargando: {url}")

    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            with open(local_path, "wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)

        return local_path

    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(f"Archivo no encontrado: {url}")
            return None
        raise

    except Exception as exc:
        print(f"No se pudo descargar {url}: {exc}")
        return None


def iter_parquet_batches(path: Path, batch_size: int) -> Iterator[pd.DataFrame]:
    """
    Lee un archivo Parquet por lotes usando PyArrow.

    Esto evita hacer:
        pd.read_parquet(path)

    porque eso cargaría todo el archivo en memoria.
    """
    parquet_file = pq.ParquetFile(path)

    for record_batch in parquet_file.iter_batches(batch_size=batch_size):
        yield record_batch.to_pandas()


# =============================================================================
# Limpieza, estandarización y filtrado
# =============================================================================

def standardize_trip_batch(
    df: pd.DataFrame,
    service: str,
    year: int,
    month: int,
    source_path: str,
    run_id: str,
) -> pd.DataFrame:
    """
    Estandariza columnas Yellow/Green para que ambas puedan entrar a un esquema común.

    Yellow usa:
        tpep_pickup_datetime
        tpep_dropoff_datetime

    Green usa:
        lpep_pickup_datetime
        lpep_dropoff_datetime

    Ambas se transforman a:
        PICKUP_DATETIME
        DROPOFF_DATETIME
    """
    df = df.copy()

    df.columns = [str(c).strip().lower() for c in df.columns]

    rename_map = {
        "vendorid": "VENDORID",
        "tpep_pickup_datetime": "PICKUP_DATETIME",
        "tpep_dropoff_datetime": "DROPOFF_DATETIME",
        "lpep_pickup_datetime": "PICKUP_DATETIME",
        "lpep_dropoff_datetime": "DROPOFF_DATETIME",
        "passenger_count": "PASSENGER_COUNT",
        "trip_distance": "TRIP_DISTANCE",
        "ratecodeid": "RATECODEID",
        "store_and_fwd_flag": "STORE_AND_FWD_FLAG",
        "pulocationid": "PULOCATIONID",
        "dolocationid": "DOLOCATIONID",
        "payment_type": "PAYMENT_TYPE",
        "fare_amount": "FARE_AMOUNT",
        "extra": "EXTRA",
        "mta_tax": "MTA_TAX",
        "tip_amount": "TIP_AMOUNT",
        "tolls_amount": "TOLLS_AMOUNT",
        "improvement_surcharge": "IMPROVEMENT_SURCHARGE",
        "total_amount": "TOTAL_AMOUNT",
        "congestion_surcharge": "CONGESTION_SURCHARGE",
        "airport_fee": "AIRPORT_FEE",
        "ehail_fee": "EHAIL_FEE",
        "trip_type": "TRIP_TYPE",
    }

    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})

    for col in RAW_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df["SERVICE_TYPE"] = service
    df["RUN_ID"] = run_id
    df["SOURCE_YEAR"] = year
    df["SOURCE_MONTH"] = month
    df["SOURCE_PATH"] = source_path
    df["INGESTED_AT_UTC"] = datetime.now(timezone.utc).replace(tzinfo=None)

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in DATETIME_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.tz_localize(None)

    df["STORE_AND_FWD_FLAG"] = df["STORE_AND_FWD_FLAG"].astype("string")
    df["SERVICE_TYPE"] = df["SERVICE_TYPE"].astype("string")
    df["RUN_ID"] = df["RUN_ID"].astype("string")
    df["SOURCE_PATH"] = df["SOURCE_PATH"].astype("string")

    return df[RAW_COLUMNS]


def compute_quality_stats(df: pd.DataFrame) -> Dict[str, int]:
    """
    Calcula métricas básicas de calidad para auditoría.
    """
    total = len(df)

    null_timestamps = int(
        df["PICKUP_DATETIME"].isna().sum()
        + df["DROPOFF_DATETIME"].isna().sum()
    )

    bad_timestamps = int(
        (
            df["PICKUP_DATETIME"].notna()
            & df["DROPOFF_DATETIME"].notna()
            & (df["PICKUP_DATETIME"] > df["DROPOFF_DATETIME"])
        ).sum()
    )

    range_violations = 0

    if "TRIP_DISTANCE" in df.columns:
        range_violations += int((df["TRIP_DISTANCE"] < 0).sum())

    if "TOTAL_AMOUNT" in df.columns:
        range_violations += int((df["TOTAL_AMOUNT"] < 0).sum())

    return {
        "row_count": int(total),
        "null_timestamps": int(null_timestamps),
        "bad_timestamps": int(bad_timestamps),
        "range_violations": int(range_violations),
        "issues_found": int(null_timestamps + bad_timestamps + range_violations),
    }


def filter_valid_trips(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica filtros básicos de calidad.

    Mantiene filas donde:
    - pickup/dropoff no son nulos
    - pickup <= dropoff
    - trip_distance >= 0, si existe
    - total_amount >= 0, si existe
    """
    mask = (
        df["PICKUP_DATETIME"].notna()
        & df["DROPOFF_DATETIME"].notna()
        & (df["PICKUP_DATETIME"] <= df["DROPOFF_DATETIME"])
    )

    if "TRIP_DISTANCE" in df.columns:
        mask &= df["TRIP_DISTANCE"].isna() | (df["TRIP_DISTANCE"] >= 0)

    if "TOTAL_AMOUNT" in df.columns:
        mask &= df["TOTAL_AMOUNT"].isna() | (df["TOTAL_AMOUNT"] >= 0)

    return df.loc[mask].copy()


# =============================================================================
# Escritura a Snowflake
# =============================================================================

def write_dataframe_to_snowflake(
    config: PipelineConfig,
    df: pd.DataFrame,
    table_name: str,
    schema: Optional[str] = None,
) -> int:
    """
    Escribe un DataFrame a Snowflake usando write_pandas.
    """
    if df.empty:
        return 0

    schema = schema or config.raw_schema

    conn = get_snowflake_connection(config, schema=schema)

    try:
        success, nchunks, nrows, output = write_pandas(
            conn=conn,
            df=df,
            table_name=table_name,
            database=config.database,
            schema=schema,
            quote_identifiers=False,
            auto_create_table=False,
            overwrite=False,
        )

        if not success:
            raise RuntimeError(f"write_pandas falló para {schema}.{table_name}: {output}")

        return int(nrows)

    finally:
        conn.close()


def delete_existing_partition(
    config: PipelineConfig,
    service: str,
    year: int,
    month: int,
) -> int:
    """
    Borra una partición mensual antes de reingestarla.
    Esto hace que la ingesta sea idempotente.
    """
    table_name = f"TRIPS_{service.upper()}"

    conn = get_snowflake_connection(config, schema=config.raw_schema)
    cursor = None

    try:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            DELETE FROM {config.database}.{config.raw_schema}.{table_name}
            WHERE SOURCE_YEAR = %s
              AND SOURCE_MONTH = %s
            """,
            (year, month),
        )

        return int(cursor.rowcount or 0)

    finally:
        if cursor is not None:
            cursor.close()
        conn.close()


def save_audit_rows(config: PipelineConfig, audit_rows: List[Dict[str, Any]]) -> None:
    """
    Guarda auditoría de ingesta.
    """
    if not audit_rows:
        return

    audit_df = pd.DataFrame(audit_rows)

    audit_df["CREATED_AT_UTC"] = datetime.now(timezone.utc).replace(tzinfo=None)

    expected_cols = [
        "RUN_ID",
        "SERVICE_TYPE",
        "SOURCE_YEAR",
        "SOURCE_MONTH",
        "STATUS",
        "ROW_COUNT",
        "ROW_COUNT_AFTER_FILTER",
        "ISSUES_FOUND",
        "NULL_TIMESTAMPS",
        "BAD_TIMESTAMPS",
        "RANGE_VIOLATIONS",
        "LOAD_TIME_SEC",
        "DELETED_PREVIOUS_ROWS",
        "FILTER_RAW_ROWS",
        "SOURCE_PATH",
        "ERROR",
        "CREATED_AT_UTC",
    ]

    audit_df = audit_df[expected_cols]

    written = write_dataframe_to_snowflake(
        config=config,
        df=audit_df,
        table_name="INGESTION_AUDIT",
        schema=config.raw_schema,
    )

    print(f"Auditoría guardada: {written:,} filas")


# =============================================================================
# Ingesta RAW
# =============================================================================

def ingest_month(
    config: PipelineConfig,
    service: str,
    year: int,
    month: int,
) -> Dict[str, Any]:
    """
    Ingesta un mes de un servicio específico.
    """
    table_name = f"TRIPS_{service.upper()}"
    source_url = f"{config.parquet_base_url}/{service}_tripdata_{year}-{month:02d}.parquet"

    audit = {
        "RUN_ID": config.run_id,
        "SERVICE_TYPE": service,
        "SOURCE_YEAR": year,
        "SOURCE_MONTH": month,
        "STATUS": "unknown",
        "ROW_COUNT": 0,
        "ROW_COUNT_AFTER_FILTER": 0,
        "ISSUES_FOUND": 0,
        "NULL_TIMESTAMPS": 0,
        "BAD_TIMESTAMPS": 0,
        "RANGE_VIOLATIONS": 0,
        "LOAD_TIME_SEC": 0.0,
        "DELETED_PREVIOUS_ROWS": 0,
        "FILTER_RAW_ROWS": config.filter_raw_rows,
        "SOURCE_PATH": source_url,
        "ERROR": None,
    }

    start = time.time()
    local_path = None

    print(f"\nProcesando {service} {year}-{month:02d}")

    try:
        local_path = download_parquet(config, service, year, month)

        if local_path is None:
            audit["STATUS"] = "missing"
            return audit

        deleted_rows = delete_existing_partition(config, service, year, month)
        audit["DELETED_PREVIOUS_ROWS"] = deleted_rows

        total_rows = 0
        total_rows_after_filter = 0
        total_issues = 0
        total_null_timestamps = 0
        total_bad_timestamps = 0
        total_range_violations = 0

        batch_number = 0

        for raw_batch in iter_parquet_batches(local_path, batch_size=config.batch_size):
            batch_number += 1

            batch = standardize_trip_batch(
                df=raw_batch,
                service=service,
                year=year,
                month=month,
                source_path=source_url,
                run_id=config.run_id,
            )

            stats = compute_quality_stats(batch)

            total_rows += stats["row_count"]
            total_issues += stats["issues_found"]
            total_null_timestamps += stats["null_timestamps"]
            total_bad_timestamps += stats["bad_timestamps"]
            total_range_violations += stats["range_violations"]

            if config.filter_raw_rows:
                batch_to_write = filter_valid_trips(batch)
            else:
                batch_to_write = batch

            rows_after_filter = len(batch_to_write)
            total_rows_after_filter += rows_after_filter

            written = write_dataframe_to_snowflake(
                config=config,
                df=batch_to_write,
                table_name=table_name,
                schema=config.raw_schema,
            )

            print(
                f"  batch={batch_number} | "
                f"leídas={len(batch):,} | "
                f"guardadas={written:,} | "
                f"issues={stats['issues_found']:,}"
            )

        audit["STATUS"] = "ok"
        audit["ROW_COUNT"] = total_rows
        audit["ROW_COUNT_AFTER_FILTER"] = total_rows_after_filter
        audit["ISSUES_FOUND"] = total_issues
        audit["NULL_TIMESTAMPS"] = total_null_timestamps
        audit["BAD_TIMESTAMPS"] = total_bad_timestamps
        audit["RANGE_VIOLATIONS"] = total_range_violations
        audit["LOAD_TIME_SEC"] = round(time.time() - start, 2)

        print(
            f"OK {service} {year}-{month:02d}: "
            f"leídas={total_rows:,}, "
            f"guardadas={total_rows_after_filter:,}, "
            f"issues={total_issues:,}, "
            f"tiempo={audit['LOAD_TIME_SEC']}s"
        )

        return audit

    except Exception as exc:
        audit["STATUS"] = "failed"
        audit["ERROR"] = str(exc)
        audit["LOAD_TIME_SEC"] = round(time.time() - start, 2)

        print(f"ERROR {service} {year}-{month:02d}: {exc}")

        return audit

    finally:
        if local_path is not None and local_path.exists():
            local_path.unlink()


def run_ingestion(config: PipelineConfig) -> List[Dict[str, Any]]:
    """
    Ejecuta la ingesta RAW para servicios, años y meses configurados.
    """
    print("\nIniciando ingesta RAW")
    print(f"Servicios: {config.services}")
    print(f"Años: {config.start_year}-{config.end_year}")
    print(f"Meses: {config.months}")
    print(f"Batch size: {config.batch_size:,}")
    print(f"Filtro RAW: {config.filter_raw_rows}")
    print(f"RUN_ID: {config.run_id}")

    audit_rows: List[Dict[str, Any]] = []

    for service in config.services:
        for year in range(config.start_year, config.end_year + 1):
            for month in config.months:
                audit = ingest_month(config, service, year, month)
                audit_rows.append(audit)

    save_audit_rows(config, audit_rows)

    ok = sum(1 for row in audit_rows if row["STATUS"] == "ok")
    missing = sum(1 for row in audit_rows if row["STATUS"] == "missing")
    failed = sum(1 for row in audit_rows if row["STATUS"] == "failed")
    rows = sum(int(row["ROW_COUNT_AFTER_FILTER"] or 0) for row in audit_rows)

    print("\nResumen ingesta RAW")
    print(f"OK: {ok}")
    print(f"Missing: {missing}")
    print(f"Failed: {failed}")
    print(f"Filas guardadas: {rows:,}")

    return audit_rows


# =============================================================================
# Catálogos y enriquecimiento
# =============================================================================

def recreate_lookup_tables(config: PipelineConfig) -> None:
    """
    Crea TAXI_ZONES y dimensiones manuales usadas para enriquecer trips.
    """
    print("\nCargando catálogos y lookups...")

    conn = get_snowflake_connection(config, schema=config.raw_schema)
    cursor = None

    try:
        cursor = conn.cursor()

        cursor.execute(
            f"""
            CREATE OR REPLACE TABLE {config.database}.{config.raw_schema}.DIM_VENDOR (
                VENDOR_ID NUMBER(38, 0),
                VENDOR_NAME VARCHAR
            )
            """
        )

        cursor.executemany(
            f"INSERT INTO {config.database}.{config.raw_schema}.DIM_VENDOR VALUES (%s, %s)",
            [
                (1, "Creative Mobile Technologies, LLC"),
                (2, "Curb Mobility, LLC"),
                (3, "Unknown"),
                (4, "Unknown"),
                (5, "Unknown"),
                (6, "Myle Technologies Inc"),
                (7, "Helix"),
            ],
        )

        cursor.execute(
            f"""
            CREATE OR REPLACE TABLE {config.database}.{config.raw_schema}.DIM_PAYMENT_TYPE (
                PAYMENT_TYPE_ID NUMBER(38, 0),
                PAYMENT_TYPE_DESC VARCHAR
            )
            """
        )

        cursor.executemany(
            f"INSERT INTO {config.database}.{config.raw_schema}.DIM_PAYMENT_TYPE VALUES (%s, %s)",
            [
                (0, "Flex Fare trip"),
                (1, "Credit card"),
                (2, "Cash"),
                (3, "No charge"),
                (4, "Dispute"),
                (5, "Unknown"),
                (6, "Voided trip"),
            ],
        )

        cursor.execute(
            f"""
            CREATE OR REPLACE TABLE {config.database}.{config.raw_schema}.DIM_RATE_CODE (
                RATE_CODE_ID NUMBER(38, 0),
                RATE_CODE_DESC VARCHAR
            )
            """
        )

        cursor.executemany(
            f"INSERT INTO {config.database}.{config.raw_schema}.DIM_RATE_CODE VALUES (%s, %s)",
            [
                (1, "Standard rate"),
                (2, "JFK"),
                (3, "Newark"),
                (4, "Nassau or Westchester"),
                (5, "Negotiated fare"),
                (6, "Group ride"),
                (99, "Null/unknown"),
            ],
        )

        cursor.execute(
            f"""
            CREATE OR REPLACE TABLE {config.database}.{config.raw_schema}.DIM_TRIP_TYPE (
                TRIP_TYPE_ID NUMBER(38, 0),
                TRIP_TYPE_DESC VARCHAR
            )
            """
        )

        cursor.executemany(
            f"INSERT INTO {config.database}.{config.raw_schema}.DIM_TRIP_TYPE VALUES (%s, %s)",
            [
                (1, "Street-hail"),
                (2, "Dispatch"),
            ],
        )

        conn.commit()

    finally:
        if cursor is not None:
            cursor.close()
        conn.close()

    print("Dimensiones manuales creadas.")

    taxi_zones_df = pd.read_csv(config.taxi_zones_url)

    taxi_zones_df.columns = [str(c).strip().upper() for c in taxi_zones_df.columns]

    rename_map = {
        "LOCATIONID": "LOCATIONID",
        "BOROUGH": "BOROUGH",
        "ZONE": "ZONE",
        "SERVICE_ZONE": "SERVICE_ZONE",
    }

    taxi_zones_df = taxi_zones_df.rename(columns=rename_map)

    expected_cols = ["LOCATIONID", "BOROUGH", "ZONE", "SERVICE_ZONE"]
    taxi_zones_df = taxi_zones_df[expected_cols]

    taxi_zones_df["LOCATIONID"] = pd.to_numeric(taxi_zones_df["LOCATIONID"], errors="coerce")

    execute_sql(
        config,
        f"""
        CREATE OR REPLACE TABLE {config.database}.{config.raw_schema}.TAXI_ZONES (
            LOCATIONID NUMBER(38, 0),
            BOROUGH VARCHAR,
            ZONE VARCHAR,
            SERVICE_ZONE VARCHAR
        )
        """,
        schema=config.raw_schema,
    )

    written = write_dataframe_to_snowflake(
        config=config,
        df=taxi_zones_df,
        table_name="TAXI_ZONES",
        schema=config.raw_schema,
    )

    print(f"TAXI_ZONES guardada: {written:,} filas")


def enriched_quality_filter(alias: str) -> str:
    return f"""
        {alias}.PICKUP_DATETIME IS NOT NULL
        AND {alias}.DROPOFF_DATETIME IS NOT NULL
        AND {alias}.PICKUP_DATETIME <= {alias}.DROPOFF_DATETIME
        AND ({alias}.TRIP_DISTANCE IS NULL OR {alias}.TRIP_DISTANCE >= 0)
        AND ({alias}.TOTAL_AMOUNT IS NULL OR {alias}.TOTAL_AMOUNT >= 0)
    """


def build_yellow_enriched_select(config: PipelineConfig) -> str:
    filters = [
        f"y.SOURCE_YEAR BETWEEN {config.start_year} AND {config.end_year}",
    ]

    if config.filter_enriched_rows:
        filters.append(enriched_quality_filter("y"))

    where_sql = " AND ".join(f"({f})" for f in filters)

    return f"""
    SELECT
        'yellow' AS SERVICE_TYPE,
        y.VENDORID AS VENDOR_ID,
        y.PICKUP_DATETIME,
        y.DROPOFF_DATETIME,
        y.PASSENGER_COUNT,
        y.TRIP_DISTANCE,
        y.RATECODEID AS RATE_CODE_ID,
        y.STORE_AND_FWD_FLAG,
        y.PULOCATIONID AS PU_LOCATION_ID,
        y.DOLOCATIONID AS DO_LOCATION_ID,
        y.PAYMENT_TYPE,
        y.FARE_AMOUNT,
        y.EXTRA,
        y.MTA_TAX,
        y.TIP_AMOUNT,
        y.TOLLS_AMOUNT,
        y.IMPROVEMENT_SURCHARGE,
        y.TOTAL_AMOUNT,
        y.CONGESTION_SURCHARGE,
        y.AIRPORT_FEE,
        CAST(NULL AS FLOAT) AS EHAIL_FEE,
        CAST(NULL AS FLOAT) AS TRIP_TYPE,
        CAST(NULL AS VARCHAR) AS TRIP_TYPE_DESC,

        tz_pu.BOROUGH AS PU_BOROUGH,
        tz_pu.ZONE AS PU_ZONE,
        tz_pu.SERVICE_ZONE AS PU_SERVICE_ZONE,

        tz_do.BOROUGH AS DO_BOROUGH,
        tz_do.ZONE AS DO_ZONE,
        tz_do.SERVICE_ZONE AS DO_SERVICE_ZONE,

        COALESCE(v.VENDOR_NAME, 'Unknown') AS VENDOR_NAME,
        COALESCE(pt.PAYMENT_TYPE_DESC, 'Not specified') AS PAYMENT_TYPE_DESC,
        COALESCE(rc.RATE_CODE_DESC, 'Not specified') AS RATE_CODE_DESC,

        DATEDIFF('minute', y.PICKUP_DATETIME, y.DROPOFF_DATETIME) AS TRIP_DURATION_MINUTES,

        CASE
            WHEN y.TRIP_DISTANCE IS NOT NULL
             AND y.TRIP_DISTANCE > 0
             AND DATEDIFF('minute', y.PICKUP_DATETIME, y.DROPOFF_DATETIME) > 0
            THEN y.TRIP_DISTANCE / (DATEDIFF('minute', y.PICKUP_DATETIME, y.DROPOFF_DATETIME) / 60.0)
            ELSE NULL
        END AS AVG_SPEED_MPH,

        y.RUN_ID,
        y.SOURCE_YEAR,
        y.SOURCE_MONTH,
        y.SOURCE_PATH,
        y.INGESTED_AT_UTC

    FROM {config.database}.{config.raw_schema}.TRIPS_YELLOW y
    LEFT JOIN {config.database}.{config.raw_schema}.TAXI_ZONES tz_pu
        ON y.PULOCATIONID = tz_pu.LOCATIONID
    LEFT JOIN {config.database}.{config.raw_schema}.TAXI_ZONES tz_do
        ON y.DOLOCATIONID = tz_do.LOCATIONID
    LEFT JOIN {config.database}.{config.raw_schema}.DIM_VENDOR v
        ON y.VENDORID = v.VENDOR_ID
    LEFT JOIN {config.database}.{config.raw_schema}.DIM_PAYMENT_TYPE pt
        ON y.PAYMENT_TYPE = pt.PAYMENT_TYPE_ID
    LEFT JOIN {config.database}.{config.raw_schema}.DIM_RATE_CODE rc
        ON y.RATECODEID = rc.RATE_CODE_ID

    WHERE {where_sql}
    """


def build_green_enriched_select(config: PipelineConfig) -> str:
    filters = [
        f"g.SOURCE_YEAR BETWEEN {config.start_year} AND {config.end_year}",
    ]

    if config.filter_enriched_rows:
        filters.append(enriched_quality_filter("g"))

    where_sql = " AND ".join(f"({f})" for f in filters)

    return f"""
    SELECT
        'green' AS SERVICE_TYPE,
        g.VENDORID AS VENDOR_ID,
        g.PICKUP_DATETIME,
        g.DROPOFF_DATETIME,
        g.PASSENGER_COUNT,
        g.TRIP_DISTANCE,
        g.RATECODEID AS RATE_CODE_ID,
        g.STORE_AND_FWD_FLAG,
        g.PULOCATIONID AS PU_LOCATION_ID,
        g.DOLOCATIONID AS DO_LOCATION_ID,
        g.PAYMENT_TYPE,
        g.FARE_AMOUNT,
        g.EXTRA,
        g.MTA_TAX,
        g.TIP_AMOUNT,
        g.TOLLS_AMOUNT,
        g.IMPROVEMENT_SURCHARGE,
        g.TOTAL_AMOUNT,
        g.CONGESTION_SURCHARGE,
        CAST(NULL AS FLOAT) AS AIRPORT_FEE,
        g.EHAIL_FEE,
        g.TRIP_TYPE,
        COALESCE(tt.TRIP_TYPE_DESC, 'Not specified') AS TRIP_TYPE_DESC,

        tz_pu.BOROUGH AS PU_BOROUGH,
        tz_pu.ZONE AS PU_ZONE,
        tz_pu.SERVICE_ZONE AS PU_SERVICE_ZONE,

        tz_do.BOROUGH AS DO_BOROUGH,
        tz_do.ZONE AS DO_ZONE,
        tz_do.SERVICE_ZONE AS DO_SERVICE_ZONE,

        COALESCE(v.VENDOR_NAME, 'Unknown') AS VENDOR_NAME,
        COALESCE(pt.PAYMENT_TYPE_DESC, 'Not specified') AS PAYMENT_TYPE_DESC,
        COALESCE(rc.RATE_CODE_DESC, 'Not specified') AS RATE_CODE_DESC,

        DATEDIFF('minute', g.PICKUP_DATETIME, g.DROPOFF_DATETIME) AS TRIP_DURATION_MINUTES,

        CASE
            WHEN g.TRIP_DISTANCE IS NOT NULL
             AND g.TRIP_DISTANCE > 0
             AND DATEDIFF('minute', g.PICKUP_DATETIME, g.DROPOFF_DATETIME) > 0
            THEN g.TRIP_DISTANCE / (DATEDIFF('minute', g.PICKUP_DATETIME, g.DROPOFF_DATETIME) / 60.0)
            ELSE NULL
        END AS AVG_SPEED_MPH,

        g.RUN_ID,
        g.SOURCE_YEAR,
        g.SOURCE_MONTH,
        g.SOURCE_PATH,
        g.INGESTED_AT_UTC

    FROM {config.database}.{config.raw_schema}.TRIPS_GREEN g
    LEFT JOIN {config.database}.{config.raw_schema}.TAXI_ZONES tz_pu
        ON g.PULOCATIONID = tz_pu.LOCATIONID
    LEFT JOIN {config.database}.{config.raw_schema}.TAXI_ZONES tz_do
        ON g.DOLOCATIONID = tz_do.LOCATIONID
    LEFT JOIN {config.database}.{config.raw_schema}.DIM_VENDOR v
        ON g.VENDORID = v.VENDOR_ID
    LEFT JOIN {config.database}.{config.raw_schema}.DIM_PAYMENT_TYPE pt
        ON g.PAYMENT_TYPE = pt.PAYMENT_TYPE_ID
    LEFT JOIN {config.database}.{config.raw_schema}.DIM_RATE_CODE rc
        ON g.RATECODEID = rc.RATE_CODE_ID
    LEFT JOIN {config.database}.{config.raw_schema}.DIM_TRIP_TYPE tt
        ON g.TRIP_TYPE = tt.TRIP_TYPE_ID

    WHERE {where_sql}
    """


def create_enriched_table(config: PipelineConfig) -> None:
    """
    Crea la tabla final enriquecida en ANALYTICS.

    La tabla se crea mediante:
    - UNION ALL de Yellow y Green
    - JOIN con zonas
    - JOIN con catálogos
    - filtros de calidad, si filter_enriched_rows=True
    """
    print("\nCreando tabla enriquecida...")

    select_blocks: List[str] = []

    if "yellow" in config.services:
        select_blocks.append(build_yellow_enriched_select(config))

    if "green" in config.services:
        select_blocks.append(build_green_enriched_select(config))

    if not select_blocks:
        raise RuntimeError("No hay servicios configurados para construir la tabla enriquecida.")

    union_sql = "\nUNION ALL\n".join(select_blocks)

    sql = f"""
    CREATE OR REPLACE TABLE {config.database}.{config.analytics_schema}.INT_TRIPS_ENRICHED AS
    {union_sql}
    """

    execute_sql(config, sql, schema=config.analytics_schema)

    print(
        f"Tabla creada: {config.database}.{config.analytics_schema}.INT_TRIPS_ENRICHED "
        f"| filter_enriched_rows={config.filter_enriched_rows}"
    )


def verify_enriched_table(config: PipelineConfig) -> None:
    """
    Verificaciones rápidas de la tabla final.
    """
    print("\nVerificando tabla enriquecida...")

    total = query_one(
        config,
        f"SELECT COUNT(*) FROM {config.database}.{config.analytics_schema}.INT_TRIPS_ENRICHED",
        schema=config.analytics_schema,
    )

    print(f"Total filas enriquecidas: {total:,}")

    conn = get_snowflake_connection(config, schema=config.analytics_schema)
    cursor = None

    try:
        cursor = conn.cursor()

        cursor.execute(
            f"""
            SELECT SERVICE_TYPE, COUNT(*) AS N
            FROM {config.database}.{config.analytics_schema}.INT_TRIPS_ENRICHED
            GROUP BY SERVICE_TYPE
            ORDER BY SERVICE_TYPE
            """
        )

        print("\nFilas por servicio:")
        for service_type, count in cursor.fetchall():
            print(f"  {service_type}: {count:,}")

        cursor.execute(
            f"""
            SELECT SOURCE_YEAR, COUNT(*) AS N
            FROM {config.database}.{config.analytics_schema}.INT_TRIPS_ENRICHED
            GROUP BY SOURCE_YEAR
            ORDER BY SOURCE_YEAR
            """
        )

        print("\nFilas por año:")
        for source_year, count in cursor.fetchall():
            print(f"  {source_year}: {count:,}")

        cursor.execute(
            f"""
            SELECT
                SUM(CASE WHEN PICKUP_DATETIME IS NULL THEN 1 ELSE 0 END) AS NULL_PICKUP,
                SUM(CASE WHEN DROPOFF_DATETIME IS NULL THEN 1 ELSE 0 END) AS NULL_DROPOFF,
                SUM(CASE WHEN PU_LOCATION_ID IS NULL THEN 1 ELSE 0 END) AS NULL_PU,
                SUM(CASE WHEN DO_LOCATION_ID IS NULL THEN 1 ELSE 0 END) AS NULL_DO,
                SUM(CASE WHEN PU_ZONE IS NULL THEN 1 ELSE 0 END) AS NULL_PU_ZONE,
                SUM(CASE WHEN DO_ZONE IS NULL THEN 1 ELSE 0 END) AS NULL_DO_ZONE,
                SUM(CASE WHEN VENDOR_NAME IS NULL THEN 1 ELSE 0 END) AS NULL_VENDOR
            FROM {config.database}.{config.analytics_schema}.INT_TRIPS_ENRICHED
            """
        )

        row = cursor.fetchone()

        print("\nNulos clave:")
        print(f"  PICKUP_DATETIME : {row[0]:,}")
        print(f"  DROPOFF_DATETIME: {row[1]:,}")
        print(f"  PU_LOCATION_ID  : {row[2]:,}")
        print(f"  DO_LOCATION_ID  : {row[3]:,}")
        print(f"  PU_ZONE         : {row[4]:,}")
        print(f"  DO_ZONE         : {row[5]:,}")
        print(f"  VENDOR_NAME     : {row[6]:,}")

    finally:
        if cursor is not None:
            cursor.close()
        conn.close()


def run_enrichment(config: PipelineConfig) -> None:
    recreate_lookup_tables(config)
    create_enriched_table(config)
    verify_enriched_table(config)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    args = parse_args()
    config = load_config(args)

    print("Configuración efectiva")
    print(f"  Database          : {config.database}")
    print(f"  Warehouse         : {config.warehouse}")
    print(f"  RAW schema        : {config.raw_schema}")
    print(f"  Analytics schema  : {config.analytics_schema}")
    print(f"  Servicios         : {', '.join(config.services)}")
    print(f"  Años              : {config.start_year}-{config.end_year}")
    print(f"  Meses             : {', '.join(str(m) for m in config.months)}")
    print(f"  Batch size        : {config.batch_size:,}")
    print(f"  RUN_ID            : {config.run_id}")
    print(f"  FILTER_RAW_ROWS   : {config.filter_raw_rows}")
    print(f"  FILTER_ENRICHED   : {config.filter_enriched_rows}")

    setup_snowflake(config)

    if not args.skip_ingest:
        run_ingestion(config)
    else:
        print("\nSaltando ingesta RAW por --skip-ingest")

    if not args.skip_enrichment:
        run_enrichment(config)
    else:
        print("\nSaltando enriquecimiento por --skip-enrichment")

    print("\nPipeline finalizado correctamente.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nPipeline falló: {exc}", file=sys.stderr)
        raise