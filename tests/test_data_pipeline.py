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
