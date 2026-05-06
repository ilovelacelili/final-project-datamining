import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

# Permitir `python src/data/ingestion.py` desde la raíz.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_conn

# La ingesta escribe en RAW con rol SYSADMIN, distinto a los demás módulos.
INGEST_SCHEMA = os.environ.get("SNOWFLAKE_SCHEMA_RAW", "RAW")
INGEST_ROLE = os.environ.get("SNOWFLAKE_ROLE", "SYSADMIN")

START_YEAR = 2015
END_YEAR = 2025
SERVICES = ['yellow', 'green']
BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
TAXI_ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
TMP_DIR = "/tmp/nyc_taxi"
RUN_ID = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

os.makedirs(TMP_DIR, exist_ok=True)

def standardize_and_enrich(df, service, year, month, source_url):
    """Homogeneiza columnas de Yellow/Green y agrega linaje."""
    df.columns = [c.lower() for c in df.columns]
    
    rename_map = {
        "tpep_pickup_datetime": "pickup_datetime",
        "tpep_dropoff_datetime": "dropoff_datetime",
        "lpep_pickup_datetime": "pickup_datetime",
        "lpep_dropoff_datetime": "dropoff_datetime",
        "vendorid": "vendorid",
        "ratecodeid": "ratecodeid",
        "pulocationid": "pulocationid",
        "dolocationid": "dolocationid"
    }
    df.rename(columns=rename_map, inplace=True)
    
    # 1. Convertir a datetime puro
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
    df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'], errors='coerce')

    # 2. Truco anti-PyArrow: Convertir a String explícitamente para Snowflake
    df['pickup_datetime'] = df['pickup_datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    df['dropoff_datetime'] = df['dropoff_datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')

    # 3. Metadatos (linaje)
    df["service_type"] = service
    df["run_id"] = RUN_ID
    df["source_year"] = year
    df["source_month"] = month
    df["source_path"] = source_url
    df["ingested_at_utc"] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')
    
    # Pasar a mayúsculas para Snowflake
    df.columns = [c.upper() for c in df.columns]
    return df

def get_quality_stats(df):
    """Calcula métricas de calidad (solo lectura en consola)."""
    null_ts = df["PICKUP_DATETIME"].isna().sum() + df["DROPOFF_DATETIME"].isna().sum()
    bad_ts = (df["PICKUP_DATETIME"] > df["DROPOFF_DATETIME"]).sum()
    
    range_viol = 0
    if "TRIP_DISTANCE" in df.columns:
        range_viol += (df["TRIP_DISTANCE"] < 0).sum()
    if "TOTAL_AMOUNT" in df.columns:
        range_viol += (df["TOTAL_AMOUNT"] < 0).sum()
        
    return int(null_ts + bad_ts + range_viol)

def load_taxi_zones(conn, db, schema):
    """Descarga e ingesta el catálogo de Taxi Zones directamente en Snowflake."""
    print(f"\nProcesando Catálogo: TAXI_ZONES...")
    start_time = time.time()
    
    try:
        # Leer el CSV directamente desde internet a memoria
        df_zones = pd.read_csv(TAXI_ZONES_URL)
        
        # Limpiar columnas
        df_zones.columns = [str(c).strip().upper() for c in df_zones.columns]
        
        # Idempotencia: Borrar tabla si ya existe para asegurar un reemplazo limpio
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS TAXI_ZONES")
        
        # Subir a Snowflake
        success, _, nrows, _ = write_pandas(
            conn=conn, 
            df=df_zones, 
            table_name="TAXI_ZONES",
            database=db, 
            schema=schema,
            auto_create_table=True, 
            quote_identifiers=False
        )
        
        if success:
            load_time = round(time.time() - start_time, 2)
            print(f"  -> OK: {nrows} zonas creadas en la tabla TAXI_ZONES | Tiempo: {load_time}s")
        else:
            print("  -> ERROR: falló write_pandas para Taxi Zones")
            
    except Exception as e:
        print(f"  -> ERROR crítico procesando Taxi Zones: {e}")

def main():
    print(f"Iniciando Pipeline de Ingesta | RUN_ID: {RUN_ID}")
    conn = get_conn(schema=INGEST_SCHEMA, role=INGEST_ROLE)
    cursor = conn.cursor()

    db = os.environ.get("SNOWFLAKE_DATABASE")
    schema = INGEST_SCHEMA
    
    # 1. Configurar contexto
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db}")
    cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {db}.{schema}")
    cursor.execute(f"USE DATABASE {db}")
    cursor.execute(f"USE SCHEMA {schema}")

    # 2. Bucle Principal de Viajes (Yellow/Green)
    for service in SERVICES:
        table_name = f"TRIPS_{service.upper()}"

        for year in range(START_YEAR, END_YEAR + 1):
            for month in range(1, 13):
                filename = f"{service}_tripdata_{year}-{month:02d}.parquet"
                url = f"{BASE_URL}/{filename}"
                local_path = f"{TMP_DIR}/{filename}"

                print(f"Procesando {service.upper()} {year}-{month:02d}...")
                start_time = time.time()

                try:
                    urllib.request.urlretrieve(url, local_path)
                except Exception:
                    print(f"  -> Archivo omitido (no encontrado o futuro).")
                    continue

                try:
                    df = pd.read_parquet(local_path)
                    df = standardize_and_enrich(df, service, year, month, url)
                    issues = get_quality_stats(df)

                    try:
                        cursor.execute(f"DELETE FROM {table_name} WHERE SOURCE_YEAR = {year} AND SOURCE_MONTH = {month}")
                    except snowflake.connector.errors.ProgrammingError:
                        pass 

                    success, _, nrows, _ = write_pandas(
                        conn=conn, df=df, table_name=table_name,
                        database=db, schema=schema,
                        auto_create_table=True, quote_identifiers=False
                    )

                    if success:
                        load_time = round(time.time() - start_time, 2)
                        print(f"  -> OK: {nrows} filas | Issues: {issues} | Tiempo: {load_time}s")
                    else:
                        raise RuntimeError("write_pandas retornó success=False")

                except Exception as e:
                    print(f"  -> ERROR crítico procesando el archivo: {e}")

                finally:
                    if os.path.exists(local_path):
                        os.remove(local_path)

    # 3. Carga final del catálogo de zonas
    load_taxi_zones(conn, db, schema)

    cursor.close()
    conn.close()
    print("\nPipeline de ingesta finalizado.")

if __name__ == "__main__":
    main()