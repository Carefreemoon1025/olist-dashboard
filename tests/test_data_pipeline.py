from pathlib import Path

import pandas as pd

from olist_copilot.data_pipeline.demo_data import generate_demo_dataset
from olist_copilot.data_pipeline.pipeline import build_warehouse


def test_demo_dataset_is_deterministic_and_has_required_tables(tmp_path: Path):
    first = generate_demo_dataset(tmp_path / "first", seed=7, n_orders=40)
    second = generate_demo_dataset(tmp_path / "second", seed=7, n_orders=40)

    assert sorted(first) == sorted(second)
    assert set(first) >= {"orders", "order_items", "products", "sellers", "customers", "reviews"}
    pd.testing.assert_frame_equal(
        pd.read_csv(first["orders"]),
        pd.read_csv(second["orders"]),
    )


def test_build_warehouse_creates_joined_analysis_mart(tmp_path: Path):
    files = generate_demo_dataset(tmp_path / "raw", seed=3, n_orders=50)
    db_path = tmp_path / "warehouse" / "olist.duckdb"

    summary = build_warehouse(files, db_path)

    assert db_path.exists()
    assert summary["fact_orders"] == 50
    assert summary["mart_order_analysis"] == 50

def test_read_source_file_supports_excel(tmp_path):
    from olist_copilot.data_pipeline.pipeline import read_source_file

    source = tmp_path / "sample.xlsx"
    pd.DataFrame({"order_id": ["1", "2"], "amount": [10, 20]}).to_excel(source, index=False)
    result = read_source_file(source)

    assert list(result.columns) == ["order_id", "amount"]
    assert result["amount"].sum() == 30


def test_build_warehouse_exports_processed_parquet(tmp_path: Path):
    files = generate_demo_dataset(tmp_path / "raw", seed=8, n_orders=20)
    build_warehouse(files, tmp_path / "warehouse" / "olist.duckdb")

    assert (tmp_path / "processed" / "mart_order_analysis.parquet").exists()

def test_source_resolution_does_not_use_partial_real_directory(tmp_path: Path):
    from olist_copilot.data_pipeline.pipeline import resolve_source_files

    raw_dir = tmp_path / "raw"
    demo_dir = tmp_path / "demo_raw"
    raw_dir.mkdir()
    demo_dir.mkdir()
    (raw_dir / "important_real_file.csv").write_text("do not overwrite", encoding="utf-8")
    demo_files = generate_demo_dataset(demo_dir, seed=11, n_orders=12)

    files, is_demo = resolve_source_files(raw_dir, demo_dir)

    assert is_demo is True
    assert files["orders"] == demo_files["orders"]
    assert (raw_dir / "important_real_file.csv").read_text(encoding="utf-8") == "do not overwrite"

def test_warehouse_keeps_item_grain_for_multi_item_orders(tmp_path: Path):
    import duckdb

    files = generate_demo_dataset(tmp_path / "raw", seed=14, n_orders=50)
    db_path = tmp_path / "warehouse" / "olist.duckdb"
    build_warehouse(files, db_path)

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        item_rows = con.execute("SELECT COUNT(*) FROM fact_order_items").fetchone()[0]
        order_rows = con.execute("SELECT COUNT(*) FROM fact_orders").fetchone()[0]
        item_total = con.execute("SELECT SUM(price + freight_value) FROM fact_order_items").fetchone()[0]
        order_total = con.execute("SELECT SUM(order_total_value + freight_value) FROM mart_order_analysis").fetchone()[0]
    finally:
        con.close()

    assert item_rows >= order_rows
    assert round(item_total, 6) == round(order_total, 6)

def test_schema_validation_reports_missing_columns(tmp_path: Path):
    from olist_copilot.data_pipeline.pipeline import validate_source_tables

    tables = {"orders": pd.DataFrame({"order_id": ["1"]})}
    try:
        validate_source_tables(tables)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected schema validation to fail")

    assert "order_status" in message
    assert "order_items" in message
