"""Reusable, grain-aware analytics queries for dashboards and the AI assistant."""
from __future__ import annotations

import pandas as pd

from olist_copilot.warehouse.connection import read_table


def load_mart(db_path: str) -> pd.DataFrame:
    return read_table(db_path, "mart_order_analysis")


def load_item_mart(db_path: str) -> pd.DataFrame:
    return read_table(db_path, "fact_order_items")


def valid_orders(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "order_status" in result.columns:
        result = result[~result["order_status"].astype(str).str.lower().isin({"canceled", "unavailable"})]
    return result


def _paid_amount_column(frame: pd.DataFrame) -> pd.Series:
    if "price" in frame.columns:
        return pd.to_numeric(frame["price"], errors="coerce").fillna(0) + pd.to_numeric(frame.get("freight_value", 0), errors="coerce").fillna(0)
    return pd.to_numeric(frame.get("order_total_value", 0), errors="coerce").fillna(0) + pd.to_numeric(frame.get("freight_value", 0), errors="coerce").fillna(0)


def monthly_trend(mart: pd.DataFrame) -> pd.DataFrame:
    frame = valid_orders(mart)
    frame["order_purchase_timestamp"] = pd.to_datetime(frame["order_purchase_timestamp"], errors="coerce")
    frame["order_month"] = frame["order_purchase_timestamp"].dt.to_period("M").astype(str)
    frame["paid_amount"] = _paid_amount_column(frame)
    result = frame.groupby("order_month", as_index=False).agg(
        order_count=("order_id", "nunique"),
        paid_amount=("paid_amount", "sum"),
        late_delivery_rate=("late_flag", "mean"),
        average_review_score=("review_score", "mean"),
    )
    return result.sort_values("order_month").reset_index(drop=True)


def ranking_by_dimension(mart: pd.DataFrame, metric: str, dimension: str, limit: int = 10) -> pd.DataFrame:
    frame = valid_orders(mart)
    if dimension not in frame.columns:
        raise ValueError(f"不支持的分析维度: {dimension}")
    if metric not in {"order_count", "paid_amount", "late_delivery_rate", "average_review_score"}:
        raise ValueError(f"不支持的指标: {metric}")
    frame = frame[frame[dimension].notna()].copy()
    if metric == "order_count":
        result = frame.groupby(dimension, as_index=False)["order_id"].nunique().rename(columns={"order_id": metric})
    elif metric == "paid_amount":
        frame[metric] = _paid_amount_column(frame)
        result = frame.groupby(dimension, as_index=False)[metric].sum()
    elif metric == "late_delivery_rate":
        result = frame.groupby(dimension, as_index=False)["late_flag"].mean().rename(columns={"late_flag": metric})
    else:
        result = frame.groupby(dimension, as_index=False)["review_score"].mean().rename(columns={"review_score": metric})
    return result.sort_values(metric, ascending=False).head(limit).reset_index(drop=True)


def delivery_review_comparison(mart: pd.DataFrame) -> pd.DataFrame:
    frame = valid_orders(mart)
    frame = frame[frame["late_flag"].notna()].copy()
    frame["delivery_group"] = frame["late_flag"].map({0.0: "正常送达", 1.0: "延迟送达"})
    return frame.groupby("delivery_group", as_index=False).agg(
        order_count=("order_id", "nunique"),
        average_review_score=("review_score", "mean"),
        average_delivery_days=("delivery_days", "mean"),
    )


def seller_performance(item_mart: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    frame = valid_orders(item_mart)
    frame["paid_amount"] = _paid_amount_column(frame)
    return (
        frame.groupby("seller_id", as_index=False)
        .agg(
            order_count=("order_id", "nunique"),
            paid_amount=("paid_amount", "sum"),
            late_delivery_rate=("late_flag", "mean"),
            average_review_score=("review_score", "mean"),
        )
        .sort_values("order_count", ascending=False)
        .head(limit)
        .reset_index(drop=True)
    )
