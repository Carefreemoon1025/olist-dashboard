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
