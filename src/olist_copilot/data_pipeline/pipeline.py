"""Data ingestion, cleaning and DuckDB warehouse construction."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping

import duckdb
import numpy as np
import pandas as pd


DATE_COLUMNS = {
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "shipping_limit_date",
}


def read_csv_tables(files: Mapping[str, Path]) -> Dict[str, pd.DataFrame]:
    """Read Olist CSV files into dataframes with normalized datetime columns."""
    tables: Dict[str, pd.DataFrame] = {}
    for name, path in files.items():
        if name == "translation" or not Path(path).exists():
            continue
        frame = pd.read_csv(path)
        frame.columns = [str(column).strip() for column in frame.columns]
        for column in DATE_COLUMNS.intersection(frame.columns):
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
        tables[name] = frame
    return tables


def _first_existing(frame: pd.DataFrame, candidates: list[str], default=None) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return frame[column]
    return pd.Series(default, index=frame.index)


def clean_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Normalize order columns and derive delivery labels without leakage."""
    result = orders.copy()
    result.columns = [str(column).strip() for column in result.columns]
    for column in DATE_COLUMNS.intersection(result.columns):
        result[column] = pd.to_datetime(result[column], errors="coerce")

    required = ["order_id", "order_status"]
    missing = [column for column in required if column not in result.columns]
    if missing:
        raise ValueError(f"orders 缺少必要字段: {', '.join(missing)}")

    result["order_status"] = result["order_status"].fillna("unknown").astype(str).str.lower()
    for column in ("order_purchase_timestamp", "order_estimated_delivery_date", "order_delivered_customer_date"):
        if column not in result.columns:
            result[column] = pd.NaT

    delivered = result["order_delivered_customer_date"]
    estimated = result["order_estimated_delivery_date"]
    valid_delivery = delivered.notna() & estimated.notna() & result["order_status"].eq("delivered")
    result["delivery_days"] = (delivered - result["order_purchase_timestamp"]).dt.total_seconds() / 86400
    result["estimated_days"] = (estimated - result["order_purchase_timestamp"]).dt.total_seconds() / 86400
    result["late_flag"] = pd.Series(np.nan, index=result.index, dtype="float64")
    result.loc[valid_delivery, "late_flag"] = (delivered[valid_delivery] > estimated[valid_delivery]).astype(float)
    return result.drop_duplicates(subset=["order_id"]).reset_index(drop=True)


def _aggregate_items(items: pd.DataFrame, products: pd.DataFrame, sellers: pd.DataFrame) -> pd.DataFrame:
    items = items.copy()
    if "price" not in items:
        items["price"] = 0.0
    if "freight_value" not in items:
        items["freight_value"] = 0.0
    items["price"] = pd.to_numeric(items["price"], errors="coerce").fillna(0.0)
    items["freight_value"] = pd.to_numeric(items["freight_value"], errors="coerce").fillna(0.0)

    enriched = items.merge(products, on="product_id", how="left", suffixes=("", "_product"))
    enriched = enriched.merge(sellers[["seller_id", "seller_state"]], on="seller_id", how="left")
    enriched["product_weight_g"] = pd.to_numeric(enriched.get("product_weight_g", 0), errors="coerce").fillna(0.0)
    enriched["item_count"] = 1
    grouped = enriched.groupby("order_id", as_index=False).agg(
        order_total_value=("price", "sum"),
        freight_value=("freight_value", "sum"),
        item_count=("item_count", "sum"),
        product_weight_g=("product_weight_g", "mean"),
        product_category=("product_category_name", "first"),
        seller_id=("seller_id", "first"),
        seller_state=("seller_state", "first"),
    )
    return grouped


def build_analysis_mart(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Join cleaned source tables into one order-level analysis mart."""
    required = {"orders", "order_items", "products", "sellers", "customers", "reviews"}
    missing = sorted(required - set(tables))
    if missing:
        raise ValueError(f"缺少 Olist 数据表: {', '.join(missing)}")

    orders = clean_orders(tables["orders"])
    customers = tables["customers"].copy()
    sellers = tables["sellers"].copy()
    products = tables["products"].copy()
    items = tables["order_items"].copy()
    reviews = tables["reviews"].copy()

    customer_columns = [column for column in ["customer_id", "customer_state", "customer_city"] if column in customers.columns]
    seller_columns = [column for column in ["seller_id", "seller_state", "seller_city"] if column in sellers.columns]
    customer_info = customers[customer_columns].drop_duplicates("customer_id")
    seller_info = sellers[seller_columns].drop_duplicates("seller_id")

    item_agg = _aggregate_items(items, products, sellers)
    review_agg = reviews.groupby("order_id", as_index=False).agg(review_score=("review_score", "mean"))
    payment_agg = pd.DataFrame(columns=["order_id", "payment_value", "payment_type"])
    if "payments" in tables:
        payments = tables["payments"].copy()
        payments["payment_value"] = pd.to_numeric(payments.get("payment_value", 0), errors="coerce").fillna(0.0)
        payment_agg = payments.groupby("order_id", as_index=False).agg(
            payment_value=("payment_value", "sum"),
            payment_type=("payment_type", "first"),
        )

    mart = orders.merge(customer_info, on="customer_id", how="left")
    mart = mart.merge(item_agg, on="order_id", how="left")
    mart = mart.merge(review_agg, on="order_id", how="left")
    mart = mart.merge(payment_agg, on="order_id", how="left")
    mart["payment_value"] = mart["payment_value"].fillna(mart["order_total_value"])
    mart["order_total_value"] = mart["order_total_value"].fillna(mart["payment_value"]).fillna(0.0)
    mart["freight_value"] = mart["freight_value"].fillna(0.0)
    mart["item_count"] = mart["item_count"].fillna(0).astype(int)
    mart["product_weight_g"] = mart["product_weight_g"].fillna(0.0)
    mart["cross_state"] = (mart["customer_state"].fillna("") != mart["seller_state"].fillna("")).astype(int)
    mart["distance_km"] = np.where(mart["cross_state"].eq(1), 500.0, 50.0)
    mart["order_month"] = pd.to_datetime(mart["order_purchase_timestamp"], errors="coerce").dt.to_period("M").astype(str)
    return mart


def _register_table(con, name: str, frame: pd.DataFrame) -> None:
    con.register(f"_{name}_df", frame)
    con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM "_{name}_df"')
    con.unregister(f"_{name}_df")


def build_warehouse(files: Mapping[str, Path], db_path: Path) -> Dict[str, int]:
    """Build a reproducible DuckDB warehouse and return table row counts."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tables = read_csv_tables(files)
    mart = build_analysis_mart(tables)
    orders = clean_orders(tables["orders"])
    con = duckdb.connect(str(db_path))
    try:
        for name, frame in tables.items():
            if name != "translation":
                _register_table(con, name, frame)
        _register_table(con, "fact_orders", orders)
        _register_table(con, "mart_order_analysis", mart)
        con.execute("CREATE OR REPLACE TABLE metadata AS SELECT CURRENT_TIMESTAMP AS built_at, ? AS source_tables", [", ".join(sorted(tables))])
        counts = {
            name: int(con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])
            for name in ["fact_orders", "mart_order_analysis"]
        }
    finally:
        con.close()
    return counts


def discover_olist_files(raw_dir: Path) -> Dict[str, Path]:
    """Discover canonical Olist filenames in a directory."""
    raw_dir = Path(raw_dir)
    aliases = {
        "orders": "olist_orders_dataset.csv",
        "order_items": "olist_order_items_dataset.csv",
        "payments": "olist_order_payments_dataset.csv",
        "reviews": "olist_order_reviews_dataset.csv",
        "products": "olist_products_dataset.csv",
        "sellers": "olist_sellers_dataset.csv",
        "customers": "olist_customers_dataset.csv",
    }
    return {name: raw_dir / filename for name, filename in aliases.items() if (raw_dir / filename).exists()}
