import os
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def _connection_params() -> dict:
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
        "dbname": os.getenv("POSTGRES_DB", "maintenance_db"),
        "user": os.getenv("POSTGRES_USER", "admin"),
        "password": os.getenv("POSTGRES_PASSWORD", "secret"),
    }


def get_connection():
    try:
        return psycopg2.connect(**_connection_params())
    except psycopg2.OperationalError as e:
        raise psycopg2.OperationalError(
            f"{e}\n\nStart Postgres first: docker compose up -d postgres"
        ) from e


def init_schema(conn=None) -> None:
    sql = _SCHEMA_PATH.read_text()
    own_conn = conn is None
    conn = conn or get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    finally:
        if own_conn:
            conn.close()


def insert_readings(df: pd.DataFrame, conn=None) -> int:
    rows = [
        (
            row.equipment_id,
            row.timestamp,
            row.vibration,
            row.temperature,
            row.pressure,
        )
        for row in df.itertuples(index=False)
    ]
    if not rows:
        return 0

    own_conn = conn is None
    conn = conn or get_connection()
    try:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO sensor_readings
                    (equipment_id, timestamp, vibration, temperature, pressure)
                VALUES %s
                """,
                rows,
            )
        conn.commit()
        return len(rows)
    finally:
        if own_conn:
            conn.close()


def ingest_from_simulator(equipment_id: str, n_hours: int = 720) -> int:
    from .sensor_simulator import SensorSimulator

    df = SensorSimulator(equipment_id).generate(n_hours=n_hours)
    return insert_readings(df)


if __name__ == "__main__":
    equipment_id = os.getenv("EQUIPMENT_ID", "PUMP-001")
    n_hours = int(os.getenv("N_HOURS", "720"))
    count = ingest_from_simulator(equipment_id, n_hours=n_hours)
    print(f"Ingested {count} readings for {equipment_id}")
