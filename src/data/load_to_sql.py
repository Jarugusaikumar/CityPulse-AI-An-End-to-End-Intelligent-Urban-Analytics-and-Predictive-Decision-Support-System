"""
Load raw CSV datasets into the CityPulse SQL database (SQLite by default).

Usage:
    python src/data/load_to_sql.py

To point at real SQL Server instead of SQLite, replace ENGINE_URL with e.g.:
    mssql+pyodbc://user:pass@server/CityPulseDB?driver=ODBC+Driver+17+for+SQL+Server
and run sql/schema.sql against that server first.
"""
import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "citypulse.db"
SCHEMA_PATH = ROOT / "sql" / "schema.sql"


def get_connection():
    if DB_PATH.exists():
        DB_PATH.unlink()
    return sqlite3.connect(DB_PATH)


def init_schema(conn):
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())
    conn.commit()


def load_locations(conn):
    zones = pd.read_csv(RAW_DIR / "traffic.csv")["zone_id"].unique()
    df = pd.DataFrame({"zone_id": zones, "zone_name": [f"Zone {z[1:]}" for z in zones]})
    df.to_sql("locations", conn, if_exists="replace", index=False)


def load_table(conn, csv_name, table_name, parse_dates=None):
    df = pd.read_csv(RAW_DIR / csv_name, parse_dates=parse_dates)
    df.to_sql(table_name, conn, if_exists="append", index=False)
    print(f"Loaded {len(df):>6} rows -> {table_name}")


def main():
    conn = get_connection()
    init_schema(conn)
    load_locations(conn)

    load_table(conn, "traffic.csv", "traffic", parse_dates=["timestamp"])
    load_table(conn, "weather.csv", "weather", parse_dates=["timestamp"])
    load_table(conn, "air_quality.csv", "air_quality", parse_dates=["timestamp"])
    load_table(conn, "energy.csv", "energy", parse_dates=["timestamp"])
    load_table(conn, "transport.csv", "transport", parse_dates=["timestamp"])
    load_table(conn, "events.csv", "events", parse_dates=["start_time", "end_time"])
    load_table(conn, "complaints.csv", "complaints", parse_dates=["timestamp"])

    # Demonstrate a SQL query (Section 10 of the spec)
    print("\nExample query - avg traffic by zone:")
    print(pd.read_sql("SELECT * FROM view_zone_avg_traffic", conn))

    conn.close()
    print(f"\nDatabase ready at {DB_PATH}")


if __name__ == "__main__":
    main()
