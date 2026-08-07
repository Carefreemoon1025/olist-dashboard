from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_DB_PATH = DATA_ROOT / "warehouse" / "olist.duckdb"
