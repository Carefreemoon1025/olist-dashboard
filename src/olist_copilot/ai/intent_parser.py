"""Controlled intent parser used before optional LLM enhancement."""
from __future__ import annotations

from typing import Any


def _base(intent: str = "unsupported") -> dict[str, Any]:
    return {
        "intent": intent,
        "metric": None,
        "dimension": None,
        "filters": {},
        "limit": 10,
        "time_range": None,
        "question": "",
    }


def parse_intent(question: str) -> dict[str, Any]:
    """Map common business questions to a small, testable intent vocabulary."""
    question = (question or "").strip()
    result = _base()
    result["question"] = question
    if not question:
        return result

    if any(word in question for word in ("股票", "基金", "天气", "写诗", "密码")):
        return result

    if "延迟率" in question and any(word in question for word in ("地区", "州", "省")):
        result.update({"intent": "ranking", "metric": "late_delivery_rate", "dimension": "customer_state"})
    elif any(word in question for word in ("订单量", "订单数")) and any(word in question for word in ("地区", "州", "省")):
        result.update({"intent": "ranking", "metric": "order_count", "dimension": "customer_state"})
    elif any(word in question for word in ("品类", "类别")) and any(word in question for word in ("销售额", "金额")):
        result.update({"intent": "ranking", "metric": "paid_amount", "dimension": "product_category"})
    elif any(word in question for word in ("卖家", "商家")) and "评价" in question:
        result.update({"intent": "seller_performance", "metric": "average_review_score", "dimension": "seller_id"})
    elif any(word in question for word in ("趋势", "变化", "最近几个月", "月度")):
        result.update({"intent": "trend", "metric": "order_count", "dimension": "order_month"})
    elif "评价" in question and any(word in question for word in ("延迟", "正常")):
        result.update({"intent": "comparison", "metric": "average_review_score", "dimension": "delivery_group"})
    elif "订单量" in question:
        result.update({"intent": "metric", "metric": "order_count", "dimension": None})
    return result
