"""Guardrails that keep AI-generated analysis inside a known metric vocabulary."""
from __future__ import annotations

from typing import Any

ALLOWED_INTENTS = {"ranking", "trend", "metric", "comparison", "seller_performance"}
ALLOWED_METRICS = {"order_count", "paid_amount", "average_order_value", "late_delivery_rate", "average_review_score"}
ALLOWED_DIMENSIONS = {"customer_state", "seller_state", "product_category", "seller_id", "order_month", "delivery_group"}


def validate_intent(intent: dict[str, Any]) -> dict[str, Any]:
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
    if not isinstance(limit, int) or not 1 <= limit <= 50:
        return {"ok": False, "reason": "limit 必须在 1 到 50 之间"}
    return {"ok": True, "reason": ""}
