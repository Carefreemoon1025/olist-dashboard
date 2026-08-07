"""Interpretable order late-delivery classification pipeline."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = ["order_total_value", "freight_value", "item_count", "product_weight_g", "distance_km", "estimated_days"]
CATEGORICAL_FEATURES = ["customer_state", "seller_state", "product_category", "payment_type", "order_month"]
POST_DELIVERY_COLUMNS = {"late_flag", "order_delivered_customer_date", "delivery_days", "review_score", "order_status"}


def build_features(mart: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build only features available before delivery; return X and y with aligned indexes."""
    if "late_flag" not in mart.columns:
        raise ValueError("mart 缺少 late_flag 标签")
    frame = mart[mart["late_flag"].notna()].copy()
    if frame["late_flag"].nunique() < 2:
        raise ValueError("订单延迟标签至少需要包含正常和延迟两类")
    for column in NUMERIC_FEATURES:
        if column not in frame.columns:
            frame[column] = 0.0
    for column in CATEGORICAL_FEATURES:
        if column not in frame.columns:
            frame[column] = "unknown"
    X = frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = frame["late_flag"].astype(int).rename("late_flag")
    return X, y


def temporal_split(
    features: pd.DataFrame,
    target: pd.Series,
    timestamps: pd.Series,
    test_size: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split chronologically so the holdout represents future orders."""
    if not 0 < test_size < 1:
        raise ValueError("test_size 必须在 0 和 1 之间")
    frame = pd.DataFrame({"timestamp": pd.to_datetime(timestamps, errors="coerce")}, index=features.index)
    frame = frame.sort_values("timestamp")
    ordered_index = frame.index
    cut = max(1, min(len(ordered_index) - 1, int(len(ordered_index) * (1 - test_size))))
    train_index, test_index = ordered_index[:cut], ordered_index[cut:]
    return features.loc[train_index], features.loc[test_index], target.loc[train_index], target.loc[test_index]


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUMERIC_FEATURES),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def train_late_delivery_model(
    features: pd.DataFrame,
    target: pd.Series,
    model_type: str = "logistic",
    test_size: float = 0.25,
    random_state: int = 42,
    timestamps: pd.Series | None = None,
) -> tuple[Pipeline, dict[str, float]]:
    """Train a baseline or random-forest model with temporal holdout when timestamps are supplied."""
    if model_type == "random_forest":
        estimator: Any = RandomForestClassifier(n_estimators=180, max_depth=8, random_state=random_state, class_weight="balanced")
    else:
        estimator = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    model = Pipeline([("preprocessor", _preprocessor()), ("estimator", estimator)])
    if timestamps is not None:
        X_train, X_test, y_train, y_test = temporal_split(features, target, timestamps, test_size)
    else:
        stratify = target if target.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=random_state, stratify=stratify)
    if y_train.nunique() < 2:
        raise ValueError("训练集至少需要包含正常和延迟两类订单")
    model.fit(X_train, y_train)
    return model, _classification_metrics(model, X_test, y_test)


def _classification_metrics(model: Pipeline, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    prediction = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1] if hasattr(model, "predict_proba") else prediction
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "recall": float(recall_score(target, prediction, zero_division=0)),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)) if len(np.unique(target)) > 1 else 0.5,
    }


def evaluate_classifier(model: Pipeline, features: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
    metrics = _classification_metrics(model, features, target)
    metrics["confusion_matrix"] = confusion_matrix(target, model.predict(features)).tolist()
    return metrics


def predict_risk(model: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    result = features.copy()
    result["risk_probability"] = model.predict_proba(features)[:, 1]
    result["risk_level"] = pd.cut(result["risk_probability"], bins=[-np.inf, 0.35, 0.65, np.inf], labels=["低风险", "中风险", "高风险"]).astype(str)
    return result
