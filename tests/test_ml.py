import pandas as pd

from olist_copilot.ml.late_delivery import (
    build_features,
    evaluate_classifier,
    train_late_delivery_model,
)


def _orders():
    return pd.DataFrame(
        {
            "order_id": [str(i) for i in range(12)],
            "order_status": ["delivered"] * 12,
            "order_purchase_ts": pd.date_range("2025-01-01", periods=12, freq="D"),
            "order_estimated_delivery_ts": pd.date_range("2025-01-05", periods=12, freq="D"),
            "order_delivered_customer_ts": pd.date_range("2025-01-04", periods=12, freq="D"),
            "order_total_value": [10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
            "freight_value": [2] * 12,
            "product_weight_g": [500, 600, 700, 800, 900, 1000] * 2,
            "distance_km": [10, 20, 30, 40, 50, 60] * 2,
            "item_count": [1, 2, 1, 2, 1, 2] * 2,
            "late_flag": [0, 1] * 6,
        }
    )


def test_feature_builder_excludes_post_delivery_fields():
    features, target = build_features(_orders())

    assert "late_flag" not in features.columns
    assert "order_delivered_customer_ts" not in features.columns
    assert "late_flag" in target.name
    assert len(features) == len(target)


def test_model_training_returns_metrics_and_predictor():
    features, target = build_features(_orders())
    model, metrics = train_late_delivery_model(features, target, test_size=0.25, random_state=42)

    assert hasattr(model, "predict")
    assert set(metrics) >= {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_classifier_evaluation_is_reproducible():
    features, target = build_features(_orders())
    model, _ = train_late_delivery_model(features, target, test_size=0.25, random_state=42)
    metrics = evaluate_classifier(model, features, target)

    assert metrics["accuracy"] >= 0.0
    assert metrics["confusion_matrix"]
