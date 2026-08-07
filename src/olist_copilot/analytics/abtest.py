"""A/B testing utilities for an independent experiment-style demo."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def proportion_test(
    conversions_a: int,
    visitors_a: int,
    conversions_b: int,
    visitors_b: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    if min(conversions_a, visitors_a, conversions_b, visitors_b) < 0:
        raise ValueError("样本量和转化数不能为负数")
    if conversions_a > visitors_a or conversions_b > visitors_b:
        raise ValueError("转化数不能大于样本量")
    if visitors_a == 0 or visitors_b == 0:
        raise ValueError("A/B 两组样本量必须大于 0")
    rate_a = conversions_a / visitors_a
    rate_b = conversions_b / visitors_b
    pooled = (conversions_a + conversions_b) / (visitors_a + visitors_b)
    standard_error = np.sqrt(pooled * (1 - pooled) * (1 / visitors_a + 1 / visitors_b))
    z_score = (rate_b - rate_a) / standard_error if standard_error else 0.0
    p_value = float(2 * (1 - stats.norm.cdf(abs(z_score)))) if standard_error else 1.0
    return {
        "rate_a": rate_a,
        "rate_b": rate_b,
        "lift_pct": (rate_b - rate_a) / rate_a * 100 if rate_a else 0.0,
        "z_score": float(z_score),
        "p_value": p_value,
        "significant": p_value < alpha,
        "conclusion": "B 组差异具有统计显著性" if p_value < alpha else "当前样本不足以证明两组存在显著差异",
    }


def mean_test(values_a: list[float], values_b: list[float], alpha: float = 0.05) -> dict[str, Any]:
    if not values_a or not values_b:
        raise ValueError("两组数据都不能为空")
    t_stat, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)
    return {
        "mean_a": float(np.mean(values_a)),
        "mean_b": float(np.mean(values_b)),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
    }
