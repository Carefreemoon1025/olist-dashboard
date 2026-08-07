"""Data ingestion, validation, multi-grain marts and DuckDB warehouse construction."""
from __future__ import annotations

import hashlib
import json
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
REQUIRED_TABLES = {"orders", "order_items", "products", "sellers", "customers", "reviews"}
REQUIRED_COLUMNS = {
    "orders": {"order_id", "customer_id", "order_status", "order_purchase_timestamp", "order_estimated_delivery_date"},
    "order_items": {"order_id", "product_id", "seller_id", "price", "freight_value"},
    "products": {"product_id", "product_category_name"},
    "sellers": {"seller_id", "seller_state"},
    "customers": {"customer_id", "customer_state"},
    "reviews": {"order_id", "review_score"},
}


def read_source_file(path: Path) -> pd.DataFrame:
    """Read CSV or Excel input with consistent column normalization."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path, engine="openpyxl")
    else:
        raise ValueError(f"暂不支持的数据文件格式: {suffix}")
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def read_csv_tables(files: Mapping[str, Path]) -> Dict[str, pd.DataFrame]:
    """Read Olist source files and normalize datetime columns."""
    tables: Dict[str, pd.DataFrame] = {}
    for name, path in files.items():
        path = Path(path)
        if not path.exists():
            continue
        frame = read_source_file(path)
        for column in DATE_COLUMNS.intersection(frame.columns):
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
        tables[name] = frame
    return tables


def validate_source_tables(tables: Mapping[str, pd.DataFrame]) -> None:
    """Validate required tables, columns and key uniqueness before building marts."""
    errors: list[str] = []
    for table in sorted(REQUIRED_TABLES - set(tables)):
        errors.append(f"缺少数据表 {table}")
    for table, required in REQUIRED_COLUMNS.items():
        if table not in tables:
            continue
        missing = sorted(required - set(tables[table].columns))
        if missing:
            errors.append(f"{table} 缺少字段: {', '.join(missing)}")
    for table, key in {"orders": "order_id", "products": "product_id", "sellers": "seller_id", "customers": "customer_id"}.items():
        if table in tables and key in tables[table].columns:
            duplicate_count = int(tables[table][key].duplicated().sum())
            if duplicate_count:
                errors.append(f"{table}.{key} 存在 {duplicate_count} 个重复值")
    if "orders" in tables and "order_purchase_timestamp" in tables["orders"].columns:
        parsed = pd.to_datetime(tables["orders"]["order_purchase_timestamp"], errors="coerce")
        if len(parsed) and parsed.notna().mean() < 0.95:
            errors.append("orders.order_purchase_timestamp 可解析比例低于 95%")
    if errors:
        raise ValueError("数据质量校验失败：" + "；".join(errors))


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
        "translation": "product_category_name_translation.csv",
    }
    return {name: raw_dir / filename for name, filename in aliases.items() if (raw_dir / filename).exists()}


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
        "translation": "product_category_name_translation.csv",
    }
    return {name: raw_dir / filename for name, filename in aliases.items() if (raw_dir / filename).exists()}

def resolve_source_files(raw_dir: Path, demo_dir: Path) -> tuple[Dict[str, Path], bool]:
    """Choose complete real data first, otherwise a separate demo directory.

    This function never writes to either directory. It prevents a partial real-data
    directory from being silently overwritten by generated demo CSVs.
    """
    raw_files = discover_olist_files(raw_dir)
    if REQUIRED_TABLES.issubset(raw_files):
        return raw_files, False
    demo_files = discover_olist_files(demo_dir)
    if REQUIRED_TABLES.issubset(demo_files):
        return demo_files, True
    raise FileNotFoundError("未找到完整 Olist 数据。请将标准 CSV 放入 data/raw，或先运行演示数据生成脚本。")


def source_fingerprint(files: Mapping[str, Path]) -> str:
    digest = hashlib.sha256()
    for name, path in sorted(files.items()):
        path = Path(path)
        digest.update(name.encode("utf-8"))
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def warehouse_is_fresh(db_path: Path, files: Mapping[str, Path]) -> bool:
    """Return whether a warehouse was built from the current source files."""
    db_path = Path(db_path)
    if not db_path.exists():
        return False
    try:
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            stored = con.execute("SELECT source_fingerprint FROM metadata LIMIT 1").fetchone()
        finally:
            con.close()
        return bool(stored and stored[0] == source_fingerprint(files))
    except Exception:
        return False


def clean_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Normalize order columns and derive delivery labels without leakage."""
    result = orders.copy()
    for column in DATE_COLUMNS.intersection(result.columns):
        result[column] = pd.to_datetime(result[column], errors="coerce")
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


def _prepare_products(products: pd.DataFrame, translation: pd.DataFrame | None = None) -> pd.DataFrame:
    result = products.copy()
    for column, default in {
        "product_category_name": "unknown",
        "product_weight_g": 0.0,
        "product_length_cm": 0.0,
        "product_height_cm": 0.0,
        "product_width_cm": 0.0,
    }.items():
        if column not in result.columns:
            result[column] = default
    result["product_category_name"] = result["product_category_name"].fillna("unknown").astype(str)
    if translation is not None and {"product_category_name", "product_category_name_english"}.issubset(translation.columns):
        result = result.merge(translation[["product_category_name", "product_category_name_english"]].drop_duplicates(), on="product_category_name", how="left")
        result["product_category"] = result["product_category_name_english"].fillna(result["product_category_name"])
    else:
        result["product_category"] = result["product_category_name"]
    return result


def build_item_mart(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Build item-grain facts so seller/category attribution remains correct."""
    orders = clean_orders(tables["orders"])
    products = _prepare_products(tables["products"], tables.get("translation"))
    items = tables["order_items"].copy()
    sellers = tables["sellers"].copy()
    customers = tables["customers"].copy()
    for column in ("price", "freight_value"):
        items[column] = pd.to_numeric(items[column], errors="coerce").fillna(0.0)
    products["product_weight_g"] = pd.to_numeric(products["product_weight_g"], errors="coerce").fillna(0.0)
    seller_info = sellers[["seller_id", "seller_state"]].drop_duplicates("seller_id")
    customer_info = customers[["customer_id", "customer_state"]].drop_duplicates("customer_id")
    result = items.merge(orders[["order_id", "customer_id", "order_status", "order_purchase_timestamp", "late_flag"]], on="order_id", how="left")
    result = result.merge(customer_info, on="customer_id", how="left")
    result = result.merge(seller_info, on="seller_id", how="left")
    result = result.merge(products[["product_id", "product_category", "product_weight_g"]], on="product_id", how="left")
    result["customer_state"] = result["customer_state"].fillna("unknown")
    result["seller_state"] = result["seller_state"].fillna("unknown")
    result["product_category"] = result["product_category"].fillna("unknown")
    result["payment_type"] = "unknown"
    result["cross_state"] = (result["customer_state"] != result["seller_state"]).astype(int)
    result["distance_km"] = np.where(result["cross_state"].eq(1), 500.0, 50.0)
    result["order_month"] = pd.to_datetime(result["order_purchase_timestamp"], errors="coerce").dt.to_period("M").astype(str)
    result["item_total_value"] = result["price"] + result["freight_value"]
    return result.reset_index(drop=True)


def build_analysis_mart(tables: Mapping[str, pd.DataFrame], item_mart: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build one row per order for KPIs and delivery prediction."""
    validate_source_tables(tables)
    orders = clean_orders(tables["orders"])
    item_mart = item_mart if item_mart is not None else build_item_mart(tables)
    item_agg = item_mart.groupby("order_id", as_index=False).agg(
        order_total_value=("price", "sum"),
        freight_value=("freight_value", "sum"),
        item_count=("order_item_id", "count"),
        product_weight_g=("product_weight_g", "mean"),
        category_count=("product_category", "nunique"),
        seller_count=("seller_id", "nunique"),
    )
    # These labels are descriptive only; category/seller rankings use item grain.
    labels = item_mart.groupby("order_id", as_index=False).agg(
        product_category=("product_category", "first"),
        seller_id=("seller_id", "first"),
        customer_state=("customer_state", "first"),
        seller_state=("seller_state", "first"),
        distance_km=("distance_km", "first"),
    )
    review_agg = tables["reviews"].groupby("order_id", as_index=False).agg(review_score=("review_score", "mean"))
    payment_agg = pd.DataFrame(columns=["order_id", "payment_value", "payment_type"])
    if "payments" in tables:
        payments = tables["payments"].copy()
        payments["payment_value"] = pd.to_numeric(payments.get("payment_value", 0), errors="coerce").fillna(0.0)
        payment_agg = payments.groupby("order_id", as_index=False).agg(payment_value=("payment_value", "sum"), payment_type=("payment_type", "first"))
    mart = orders.merge(item_agg, on="order_id", how="left").merge(labels, on="order_id", how="left").merge(review_agg, on="order_id", how="left").merge(payment_agg, on="order_id", how="left")
    mart["payment_value"] = mart["payment_value"].fillna(mart["order_total_value"])
    mart["order_total_value"] = mart["order_total_value"].fillna(mart["payment_value"]).fillna(0.0)
    mart["freight_value"] = mart["freight_value"].fillna(0.0)
    mart["item_count"] = mart["item_count"].fillna(0).astype(int)
    mart["product_weight_g"] = mart["product_weight_g"].fillna(0.0)
    for column in ("customer_state", "seller_state", "product_category", "payment_type"):
        mart[column] = mart[column].fillna("unknown")
    mart["cross_state"] = (mart["customer_state"] != mart["seller_state"]).astype(int)
    default_distance = pd.Series(np.where(mart["cross_state"].eq(1), 500.0, 50.0), index=mart.index)
    mart["distance_km"] = pd.to_numeric(mart["distance_km"], errors="coerce").fillna(default_distance)
    mart["order_month"] = pd.to_datetime(mart["order_purchase_timestamp"], errors="coerce").dt.to_period("M").astype(str)
    return mart.reset_index(drop=True)


def _register_table(con, name: str, frame: pd.DataFrame) -> None:
    con.register(f"_{name}_df", frame)
    con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM "_{name}_df"')
    con.unregister(f"_{name}_df")


def export_processed_tables(tables: Mapping[str, pd.DataFrame], orders: pd.DataFrame, item_mart: pd.DataFrame, mart: pd.DataFrame, output_dir: Path) -> None:
    """Persist cleaned source tables and both analysis grains as Parquet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)
    orders.to_parquet(output_dir / "fact_orders.parquet", index=False)
    item_mart.to_parquet(output_dir / "fact_order_items.parquet", index=False)
    mart.to_parquet(output_dir / "mart_order_analysis.parquet", index=False)


def build_warehouse(files: Mapping[str, Path], db_path: Path) -> Dict[str, int]:
    """Build a reproducible DuckDB warehouse and return key row counts."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tables = read_csv_tables(files)
    validate_source_tables(tables)
    orders = clean_orders(tables["orders"])
    item_mart = build_item_mart(tables)
    mart = build_analysis_mart(tables, item_mart)
    export_processed_tables(tables, orders, item_mart, mart, db_path.parent.parent / "processed")
    con = duckdb.connect(str(db_path))
    try:
        for name, frame in tables.items():
            _register_table(con, name, frame)
        _register_table(con, "fact_orders", orders)
        _register_table(con, "fact_order_items", item_mart)
        _register_table(con, "mart_order_analysis", mart)
        fingerprint = source_fingerprint(files)
        con.execute(
            "CREATE OR REPLACE TABLE metadata AS SELECT CURRENT_TIMESTAMP AS built_at, ? AS source_tables, ? AS source_fingerprint",
            [", ".join(sorted(files)), fingerprint],
        )
        counts = {name: int(con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]) for name in ["fact_orders", "fact_order_items", "mart_order_analysis"]}
    finally:
        con.close()
    return counts
