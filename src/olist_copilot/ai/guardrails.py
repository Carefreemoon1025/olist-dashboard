"""Guardrails that keep AI-generated analysis inside a known metric vocabulary."""
from __future__ import annotations

from typing import Any

ALLOWED_INTENTS = {"ranking", "trend", "metric", "comparison", "seller_performance"}
ALLOWED_METRICS = {"order_count", "paid_amount", "average_order_value", "late_delivery_rate", "average_review_score"}
ALLOWED_DIMENSIONS = {"customer_state", "seller_state", "product_category", "seller_id", "order_month", "delivery_group"}
ALLOWED_KEYS = {"intent", "metric", "dimension", "filters", "limit", "time_range", "question"}
ALLOWED_FILTERS = {"customer_state", "seller_state", "product_category", "order_month", "order_status"}


def validate_intent(intent: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(intent, dict):
        return {"ok": False, "reason": "intent 必须是对象"}
    extra = set(intent) - ALLOWED_KEYS
    if extra:
        return {"ok": False, "reason": f"intent 包含未允许字段: {', '.join(sorted(extra))}"}
    if intent.get("intent") == "unsupported":
        return {"ok": False, "reason": "当前版本不支持该问题"}
    if intent.get("intent") not in ALLOWED_INTENTS:
        return {"ok": False, "reason": "intent 不在允许范围内"}
    if intent.get("metric") not in ALLOWED_METRICS:
        return {"ok": False, "reason": "metric 不在允许范围内"}
    dimension = intent.get("dimension")
    if dimension is not None and dimension not in ALLOWED_DIMENSIONS:
        return {"ok": False, "reason": "dimension 不在允许范围内"}
    limit = intent.get("limit", 10)
    if type(limit) is not int or not 1 <= limit <= 50:
        return {"ok": False, "reason": "limit 必须是 1 到 50 之间的整数"}
    filters = intent.get("filters", {})
    if not isinstance(filters, dict) or set(filters) - ALLOWED_FILTERS:
        return {"ok": False, "reason": "filters 包含未允许字段"}
    if intent.get("time_range") is not None and not isinstance(intent["time_range"], (list, tuple)):
        return {"ok": False, "reason": "time_range 必须是列表或元组"}
    return {"ok": True, "reason": ""}
