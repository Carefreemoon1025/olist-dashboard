"""Optional OpenAI-compatible LLM adapter with a deterministic local fallback."""
from __future__ import annotations

import os
from typing import Any


def _client():
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=os.getenv("LLM_BASE_URL") or None)


def generate_insight(question: str, result: list[dict[str, Any]], metric_name: str) -> str:
    """Generate evidence-first insight; works without an API key."""
    client = _client()
    if client is not None:
        model = os.getenv("LLM_MODEL", "deepseek-chat")
        prompt = (
            "你是一名电商数据分析师。只能基于提供的数据回答，不要编造数字。"
            "请按‘核心发现、数据证据、可能原因、业务建议、局限性’输出。\n"
            f"问题：{question}\n指标：{metric_name}\n数据：{result}"
        )
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你负责生成可追溯的电商分析结论。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content or "模型未返回内容。"
        except Exception as exc:
            return f"大模型暂不可用，已切换到本地模板分析。原因：{type(exc).__name__}。\n\n{_fallback_insight(question, result, metric_name)}"
    return _fallback_insight(question, result, metric_name)


def _fallback_insight(question: str, result: list[dict[str, Any]], metric_name: str) -> str:
    if not result:
        return "没有找到满足条件的数据。建议检查筛选条件或数据范围。"
    first = result[0]
    key, value = next(iter(first.items()))
    numeric = next((item for name, item in first.items() if name != key and isinstance(item, (int, float))), None)
    evidence = f"{key}={value}"
    if numeric is not None:
        evidence += f"，{metric_name}={numeric:.4f}"
    return (
        f"**核心发现**\n问题‘{question}’的结果显示，排名第一的是 {key}={value}。\n\n"
        f"**数据证据**\n{evidence}。\n\n"
        "**可能原因**\n该维度在当前样本中的业务量或履约表现相对突出，需要结合订单量和时间趋势进一步判断。\n\n"
        "**业务建议**\n优先查看该维度下的卖家、品类和物流表现，并结合历史趋势制定改进措施。\n\n"
        "**局限性**\n当前结论仅基于项目数据和已定义指标，不代表因果关系。"
    )
