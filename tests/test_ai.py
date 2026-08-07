from olist_copilot.ai.intent_parser import parse_intent
from olist_copilot.ai.guardrails import validate_intent


def test_parser_maps_supported_business_question_to_structured_intent():
    intent = parse_intent("哪些地区的订单延迟率最高？")

    assert intent["intent"] == "ranking"
    assert intent["metric"] == "late_delivery_rate"
    assert intent["dimension"] == "customer_state"
    assert intent["limit"] == 10


def test_parser_does_not_claim_support_for_unknown_question():
    intent = parse_intent("请帮我预测明年股票价格")

    assert intent["intent"] == "unsupported"
    assert intent["metric"] is None


def test_guardrails_reject_unknown_metric_and_dimension():
    valid = {"intent": "ranking", "metric": "order_count", "dimension": "customer_state", "limit": 10}
    assert validate_intent(valid)["ok"] is True

    invalid = {"intent": "ranking", "metric": "drop_table", "dimension": "customer_state", "limit": 10}
    result = validate_intent(invalid)
    assert result["ok"] is False
    assert "metric" in result["reason"]

def test_guardrails_reject_boolean_limit_and_extra_fields():
    from olist_copilot.ai.guardrails import validate_intent

    boolean_limit = {"intent": "ranking", "metric": "order_count", "dimension": "customer_state", "limit": True}
    extra_field = {"intent": "ranking", "metric": "order_count", "dimension": "customer_state", "limit": 10, "sql": "DROP TABLE"}

    assert validate_intent(boolean_limit)["ok"] is False
    assert validate_intent(extra_field)["ok"] is False

def test_local_insight_fallback_includes_top_result_evidence(monkeypatch):
    from olist_copilot.ai.llm_client import generate_insight

    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = [{"customer_state": "SP", "late_delivery_rate": 0.42}]

    text = generate_insight("哪些地区延迟率最高？", result, "订单延迟率")

    assert "SP" in text
    assert "0.4200" in text
