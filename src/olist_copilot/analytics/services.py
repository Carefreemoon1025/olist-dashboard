"""Reusable analytics queries for dashboards and the AI assistant."""
from __future__ import annotations

from olist_copilot.warehouse.connection import read_table
import pandas as pd


def load_mart(db_path: str) -> pd.DataFrame:
    return read_table(db_path, "mart_order_analysis")

def monthly_trend(mart: pd.DataFrame) -> pd.DataFrame:
    frame = mart.copy()
    frame["order_purchase_timestamp"] = pd.to_datetime(frame["order_purchase_timestamp"], errors="coerce")
    frame["order_month"] = frame["order_purchase_timestamp"].dt.to_period("M").astype(str)
    result = frame.groupby("order_month", as_index=False).agg(
        order_count=("order_id", "nunique"),
        paid_amount=("order_total_value", "sum"),
        late_delivery_rate=("late_flag", "mean"),
        average_review_score=("review_score", "mean"),
    )
    return result.sort_values("order_month").reset_index(drop=True)


def ranking_by_dimension(mart: pd.DataFrame, metric: str, dimension: str, limit: int = 10) -> pd.DataFrame:
    if dimension not in mart.columns:
        raise ValueError(f"不支持的分析维度: {dimension}")
    if metric not in {"order_count", "paid_amount", "late_delivery_rate", "average_review_score"}:
        raise ValueError(f"不支持的指标: {metric}")
    grouped = mart[mart[dimension].notna()].groupby(dimension, as_index=False)
    if metric == "order_count":
        result = grouped["order_id"].nunique().rename(columns={"order_id": metric})
    elif metric == "paid_amount":
        result = grouped["order_total_value"].sum().rename(columns={"order_total_value": metric})
    elif metric == "late_delivery_rate":
        result = grouped["late_flag"].mean().rename(columns={"late_flag": metric})
    else:
        result = grouped["review_score"].mean().rename(columns={"review_score": metric})
    return result.sort_values(metric, ascending=False).head(limit).reset_index(drop=True)


def delivery_review_comparison(mart: pd.DataFrame) -> pd.DataFrame:
    frame = mart[mart["late_flag"].notna()].copy()
    frame["delivery_group"] = frame["late_flag"].map({0.0: "正常送达", 1.0: "延迟送达"})
    return frame.groupby("delivery_group", as_index=False).agg(
        order_count=("order_id", "nunique"),
        average_review_score=("review_score", "mean"),
        average_delivery_days=("delivery_days", "mean"),
    )


def seller_performance(mart: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    return (
        mart.groupby("seller_id", as_index=False)
        .agg(
            order_count=("order_id", "nunique"),
            paid_amount=("order_total_value", "sum"),
            late_delivery_rate=("late_flag", "mean"),
            average_review_score=("review_score", "mean"),
        )
        .sort_values("order_count", ascending=False)
        .head(limit)
        .reset_index(drop=True)
    )
