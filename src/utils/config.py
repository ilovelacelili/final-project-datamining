"""
Configuración central y conexión a Snowflake.

Single source of truth: cualquier módulo de src/ que necesite hablar con
Snowflake debe importar `get_conn` o `query` desde aquí, no replicar el
boilerplate.
"""
import os
from typing import Optional

import pandas as pd
import snowflake.connector
from dotenv import load_dotenv

load_dotenv()


def get_snowflake_credentials() -> dict:
    """Diccionario de credenciales desde las variables de entorno."""
    return {
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "database": os.getenv("SNOWFLAKE_DATABASE"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA"),
        "role": os.getenv("SNOWFLAKE_ROLE"),
    }


def get_conn(schema: Optional[str] = None, role: Optional[str] = None):
    """
    Abre una conexión a Snowflake.

    schema / role: overrides opcionales. La ingesta usa RAW + SYSADMIN; el
    entrenamiento y la API usan los defaults del .env (ANALYTICS).
    """
    creds = get_snowflake_credentials()
    if schema is not None:
        creds["schema"] = schema
    if role is not None:
        creds["role"] = role
    creds = {k: v for k, v in creds.items() if v is not None}
    return snowflake.connector.connect(**creds)


def query(sql: str, schema: Optional[str] = None, role: Optional[str] = None) -> pd.DataFrame:
    """Ejecuta un SELECT y devuelve un DataFrame. Cierra la conexión al terminar."""
    conn = get_conn(schema=schema, role=role)
    try:
        return pd.read_sql(sql, conn)
    finally:
        conn.close()
