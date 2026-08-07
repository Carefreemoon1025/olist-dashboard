"""Business metric definitions and calculations."""
from __future__ import annotations

from typing import Any

import pandas as pd


def metric_definitions() -> dict[str, dict[str, str]]:
    return {
        "order_count": {
            "name": "订单量",
            "definition": "有效订单数量，按 order_id 去重。",
            "formula": "COUNT(DISTINCT order_id)",
        },
        "paid_amount": {
            "name": "支付金额",
            "definition": "订单商品金额与运费金额之和。",
            "formula": "SUM(order_total_value + freight_value)",
        },
        "average_order_value": {
            "name": "客单价",
            "definition": "支付金额除以有效订单数量。",
            "formula": "paid_amount / order_count",
        },
        "late_delivery_rate": {
            "name": "订单延迟率",
            "definition": "late_flag 不为空订单中的延迟订单占比；late_flag 只对已送达订单计算。",
            "formula": "AVG(late_flag)",
        },
        "average_delivery_days": {
            "name": "平均配送天数",
            "definition": "已送达订单从下单到送达的平均天数。",
            "formula": "AVG(delivery_days)",
        },
        "average_review_score": {
            "name": "平均评价分",
            "definition": "订单评价分数的平均值。",
            "formula": "AVG(review_score)",
        },
    }


def _sum(frame: pd.DataFrame, column: str) -> float:
    values = frame[column] if column in frame.columns else pd.Series(0.0, index=frame.index)
    return float(pd.to_numeric(values, errors="coerce").fillna(0).sum())

def calculate_kpis(orders: pd.DataFrame, reviews: pd.DataFrame | None = None) -> dict[str, Any]:
    """Calculate the canonical KPI set using the documented metric definitions."""
    valid_orders = orders.copy()
    if "order_status" in valid_orders.columns:
        valid_orders = valid_orders[
            ~valid_orders["order_status"].astype(str).str.lower().isin({"canceled", "unavailable"})
        ]
    order_count = int(valid_orders["order_id"].nunique()) if "order_id" in valid_orders else int(len(valid_orders))
    paid_amount = _sum(valid_orders, "order_total_value") + _sum(valid_orders, "freight_value")
    valid_late = pd.to_numeric(valid_orders.get("late_flag", pd.Series(dtype=float)), errors="coerce").dropna()
    valid_delivery_days = pd.to_numeric(valid_orders.get("delivery_days", pd.Series(dtype=float)), errors="coerce").dropna()
    review_values = (
        reviews["review_score"]
        if reviews is not None and "review_score" in reviews
        else valid_orders.get("review_score", pd.Series(dtype=float))
    )
    review_values = pd.to_numeric(review_values, errors="coerce").dropna()
    return {
        "order_count": order_count,
        "paid_amount": paid_amount,
        "average_order_value": paid_amount / order_count if order_count else 0.0,
        "late_delivery_rate": float(valid_late.mean()) if len(valid_late) else 0.0,
        "average_delivery_days": float(valid_delivery_days.mean()) if len(valid_delivery_days) else 0.0,
        "average_review_score": float(review_values.mean()) if len(review_values) else 0.0,
    }

def add_display_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "order_purchase_timestamp" in result:
        result["order_purchase_timestamp"] = pd.to_datetime(result["order_purchase_timestamp"], errors="coerce")
        result["order_month"] = result["order_purchase_timestamp"].dt.to_period("M").astype(str)
    return result
