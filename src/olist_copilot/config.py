from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_ROOT = DATA_ROOT / "raw"
DEMO_DATA_ROOT = DATA_ROOT / "demo_raw"
DEFAULT_DB_PATH = DATA_ROOT / "warehouse" / "olist.duckdb"
DEMO_DB_PATH = DATA_ROOT / "warehouse" / "demo_olist.duckdb"
