"""Intent-to-analysis dispatcher used by the Streamlit page."""
from __future__ import annotations

from typing import Any

import pandas as pd

from olist_copilot.analytics.services import delivery_review_comparison, monthly_trend, ranking_by_dimension, seller_performance
from olist_copilot.ai.guardrails import validate_intent
from olist_copilot.ai.intent_parser import parse_intent
from olist_copilot.metrics.calculator import calculate_kpis, metric_definitions


def answer_question(mart: pd.DataFrame, question: str, item_mart: pd.DataFrame | None = None) -> dict[str, Any]:
    intent = parse_intent(question)
    validation = validate_intent(intent)
    if not validation["ok"]:
        return {"intent": intent, "validation": validation, "table": pd.DataFrame(), "insight": "当前版本暂不支持该问题。"}

    item_mart = item_mart if item_mart is not None else mart
    if intent["intent"] == "trend":
        table = monthly_trend(mart)
    elif intent["intent"] == "seller_performance":
        table = seller_performance(item_mart, limit=intent["limit"])
    elif intent["intent"] == "comparison":
        table = delivery_review_comparison(mart)
    elif intent["intent"] == "metric":
        table = pd.DataFrame([{"metric": intent["metric"], "value": _metric_value(mart, intent["metric"])}])
    else:
        source = item_mart if intent["dimension"] in {"product_category", "seller_id"} else mart
        table = ranking_by_dimension(source, intent["metric"], intent["dimension"], intent["limit"])

    return {
        "intent": intent,
        "validation": validation,
        "table": table,
        "metric_name": metric_definitions().get(intent["metric"], {}).get("name", intent["metric"]),
    }


def _metric_value(mart: pd.DataFrame, metric: str) -> float:
    kpis = calculate_kpis(mart)
    mapping = {
        "order_count": "order_count",
        "paid_amount": "paid_amount",
        "average_order_value": "average_order_value",
        "late_delivery_rate": "late_delivery_rate",
        "average_review_score": "average_review_score",
    }
    if metric not in mapping:
        raise ValueError(f"不支持的指标: {metric}")
    return float(kpis[mapping[metric]])
