"""Intent-to-analysis dispatcher used by the Streamlit page."""
from __future__ import annotations

from typing import Any

import pandas as pd

from olist_copilot.analytics.services import delivery_review_comparison, monthly_trend, ranking_by_dimension, seller_performance
from olist_copilot.ai.guardrails import validate_intent
from olist_copilot.ai.intent_parser import parse_intent
from olist_copilot.metrics.calculator import metric_definitions


def answer_question(mart: pd.DataFrame, question: str) -> dict[str, Any]:
    intent = parse_intent(question)
    validation = validate_intent(intent)
    if not validation["ok"]:
        return {"intent": intent, "validation": validation, "table": pd.DataFrame(), "insight": "当前版本暂不支持该问题。"}

    if intent["intent"] == "trend":
        table = monthly_trend(mart)
    elif intent["intent"] == "seller_performance":
        table = seller_performance(mart, limit=intent["limit"])
    elif intent["intent"] == "comparison":
        table = delivery_review_comparison(mart)
    elif intent["intent"] == "metric":
        table = pd.DataFrame([{"metric": intent["metric"], "value": _metric_value(mart, intent["metric"])}])
    else:
        table = ranking_by_dimension(mart, intent["metric"], intent["dimension"], intent["limit"])

    return {
        "intent": intent,
        "validation": validation,
        "table": table,
        "metric_name": metric_definitions().get(intent["metric"], {}).get("name", intent["metric"]),
    }


def _metric_value(mart: pd.DataFrame, metric: str) -> float:
    if metric == "order_count":
        return float(mart["order_id"].nunique())
    if metric == "paid_amount":
        return float(mart["order_total_value"].sum())
    if metric == "average_order_value":
        return float(mart["order_total_value"].sum() / mart["order_id"].nunique())
    if metric == "late_delivery_rate":
        return float(mart["late_flag"].dropna().mean())
    if metric == "average_review_score":
        return float(mart["review_score"].dropna().mean())
    raise ValueError(f"不支持的指标: {metric}")
