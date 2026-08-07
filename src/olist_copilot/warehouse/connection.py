"""DuckDB connection helpers."""
from __future__ import annotations

from pathlib import Path

import duckdb


def connect_readonly(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    """Open a read-only analytics connection for dashboards and AI tools."""
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"DuckDB 数据库不存在: {path}")
    return duckdb.connect(str(path), read_only=True)


def read_table(db_path: str | Path, table_name: str):
    """Read a table into a pandas DataFrame using a read-only connection."""
    if not table_name.replace("_", "").isalnum():
        raise ValueError("table_name 只允许字母、数字和下划线")
    con = connect_readonly(db_path)
    try:
        return con.execute(f'SELECT * FROM "{table_name}"').df()
    finally:
        con.close()
