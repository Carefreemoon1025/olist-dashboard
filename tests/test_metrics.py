import pandas as pd

from olist_copilot.metrics.calculator import calculate_kpis, metric_definitions


def test_metric_definitions_explain_business_calculation():
    definitions = metric_definitions()

    assert "late_delivery_rate" in definitions
    assert "订单延迟率" in definitions["late_delivery_rate"]["name"]
    assert "late_flag" in definitions["late_delivery_rate"]["definition"]


def test_calculate_kpis_uses_valid_delivered_orders_for_late_rate():
    orders = pd.DataFrame(
        {
            "order_id": ["1", "2", "3", "4"],
            "order_status": ["delivered", "delivered", "canceled", "delivered"],
            "order_total_value": [100.0, 200.0, 50.0, 300.0],
            "late_flag": [0, 1, None, 0],
            "delivery_days": [3.0, 8.0, None, 4.0],
        }
    )
    reviews = pd.DataFrame({"review_score": [5, 2, 4]})

    result = calculate_kpis(orders, reviews)

    assert result["order_count"] == 3
    assert result["paid_amount"] == 600.0
    assert result["average_order_value"] == 200.0
    assert round(result["late_delivery_rate"], 4) == round(1 / 3, 4)
    assert result["average_delivery_days"] == 5.0
    assert round(result["average_review_score"], 4) == round(11 / 3, 4)
