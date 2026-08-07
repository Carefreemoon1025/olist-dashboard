"""Streamlit application for the Olist E-commerce Analytics Copilot."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from olist_copilot.ai.llm_client import generate_insight
from olist_copilot.ai.query_planner import answer_question
from olist_copilot.analytics.abtest import proportion_test
from olist_copilot.analytics.services import delivery_review_comparison, monthly_trend, ranking_by_dimension, seller_performance
from olist_copilot.config import DATA_ROOT, DEFAULT_DB_PATH
from olist_copilot.data_pipeline.demo_data import generate_demo_dataset
from olist_copilot.data_pipeline.pipeline import build_warehouse, discover_olist_files
from olist_copilot.metrics.calculator import calculate_kpis
from olist_copilot.ml.late_delivery import build_features, evaluate_classifier, train_late_delivery_model


st.set_page_config(page_title="Olist 电商经营分析 Copilot", page_icon="📈", layout="wide")


@st.cache_resource(show_spinner="正在准备演示数据和 DuckDB 仓库...")
def ensure_warehouse() -> str:
    raw_dir = DATA_ROOT / "raw"
    files = discover_olist_files(raw_dir)
    required = {"orders", "order_items", "products", "sellers", "customers", "reviews"}
    if not required.issubset(files):
        generate_demo_dataset(raw_dir, seed=42, n_orders=240)
        files = discover_olist_files(raw_dir)
    if not DEFAULT_DB_PATH.exists():
        build_warehouse(files, DEFAULT_DB_PATH)
    return str(DEFAULT_DB_PATH)


@st.cache_data(show_spinner=False)
def load_data(db_path: str) -> pd.DataFrame:
    from olist_copilot.analytics.services import load_mart

    frame = load_mart(db_path)
    frame["order_purchase_timestamp"] = pd.to_datetime(frame["order_purchase_timestamp"], errors="coerce")
    return frame


@st.cache_resource(show_spinner="正在训练延迟风险模型...")
def train_model(mart: pd.DataFrame):
    features, target = build_features(mart)
    model, metrics = train_late_delivery_model(features, target, model_type="logistic")
    return model, metrics, features, target


def money(value: float) -> str:
    return f"R$ {value:,.0f}"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_sidebar(mart: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("筛选条件")
    min_date = mart["order_purchase_timestamp"].min().date()
    max_date = mart["order_purchase_timestamp"].max().date()
    date_range = st.sidebar.date_input("订单日期", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
    else:
        start = end = date_range
    states = sorted(mart["customer_state"].dropna().unique().tolist())
    selected_states = st.sidebar.multiselect("用户州", states, default=states)
    categories = sorted(mart["product_category"].dropna().unique().tolist())
    selected_categories = st.sidebar.multiselect("商品品类", categories, default=categories)
    filtered = mart[
        (mart["order_purchase_timestamp"].dt.date >= start)
        & (mart["order_purchase_timestamp"].dt.date <= end)
        & mart["customer_state"].isin(selected_states)
        & mart["product_category"].isin(selected_categories)
    ].copy()
    st.sidebar.caption(f"当前筛选：{len(filtered):,} 条订单明细")
    st.sidebar.divider()
    st.sidebar.markdown("**运行模式**")
    st.sidebar.caption("默认使用可复现的 Olist 演示数据。将真实 Olist CSV 放入 data/raw/ 后重新构建仓库即可替换数据。")
    return filtered


def overview_page(mart: pd.DataFrame):
    st.subheader("经营总览")
    kpis = calculate_kpis(mart)
    columns = st.columns(6)
    columns[0].metric("订单量", f"{kpis['order_count']:,}")
    columns[1].metric("支付金额", money(kpis["paid_amount"]))
    columns[2].metric("客单价", money(kpis["average_order_value"]))
    columns[3].metric("延迟率", pct(kpis["late_delivery_rate"]))
    columns[4].metric("平均配送", f"{kpis['average_delivery_days']:.1f} 天")
    columns[5].metric("平均评价", f"{kpis['average_review_score']:.2f}")

    trend = monthly_trend(mart)
    left, right = st.columns(2)
    with left:
        fig = px.line(trend, x="order_month", y="order_count", markers=True, title="月度订单趋势", labels={"order_month": "月份", "order_count": "订单量"})
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.bar(trend, x="order_month", y="paid_amount", title="月度支付金额", labels={"order_month": "月份", "paid_amount": "支付金额"})
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        category = ranking_by_dimension(mart, "paid_amount", "product_category", 10)
        fig = px.bar(category, x="paid_amount", y="product_category", orientation="h", title="品类支付金额排名", labels={"paid_amount": "支付金额", "product_category": "品类"})
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    with right:
        states = ranking_by_dimension(mart, "late_delivery_rate", "customer_state", 10)
        states["late_delivery_rate"] = states["late_delivery_rate"] * 100
        fig = px.bar(states, x="customer_state", y="late_delivery_rate", title="地区订单延迟率", labels={"customer_state": "用户州", "late_delivery_rate": "延迟率（%）"})
        st.plotly_chart(fig, use_container_width=True)


def product_seller_page(mart: pd.DataFrame):
    st.subheader("商品与卖家分析")
    left, right = st.columns(2)
    with left:
        category = ranking_by_dimension(mart, "order_count", "product_category", 10)
        st.markdown("#### 品类订单量")
        st.dataframe(category, use_container_width=True, hide_index=True)
    with right:
        sellers = seller_performance(mart, 15)
        st.markdown("#### 卖家表现")
        st.dataframe(sellers, use_container_width=True, hide_index=True)

    st.markdown("#### 高订单量但低评价卖家")
    sellers = seller_performance(mart, 30)
    if len(sellers) > 0:
        threshold = sellers["order_count"].quantile(0.6)
        risk_sellers = sellers[(sellers["order_count"] >= threshold) & (sellers["average_review_score"] < sellers["average_review_score"].median())]
        st.dataframe(risk_sellers, use_container_width=True, hide_index=True)


def delivery_page(mart: pd.DataFrame):
    st.subheader("履约分析与延迟风险预测")
    comparison = delivery_review_comparison(mart)
    left, right = st.columns(2)
    with left:
        st.markdown("#### 延迟与评价关系")
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        fig = px.bar(comparison, x="delivery_group", y="average_review_score", title="不同履约状态的平均评价", labels={"delivery_group": "履约状态", "average_review_score": "平均评价"})
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### 模型评估")
        try:
            model, metrics, features, target = train_model(mart)
            metric_frame = pd.DataFrame({"指标": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], "值": [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"], metrics["roc_auc"]]})
            st.dataframe(metric_frame, use_container_width=True, hide_index=True)
            st.caption("模型仅使用订单早期可获得特征，避免使用实际送达时间和评价结果造成数据泄漏。")
        except ValueError as exc:
            st.warning(str(exc))
            return

    st.markdown("#### 单笔订单风险演示")
    try:
        model, _, features, target = train_model(mart)
        sample = features.head(1)
        probability = float(model.predict_proba(sample)[:, 1][0])
        level = "高风险" if probability >= 0.65 else "中风险" if probability >= 0.35 else "低风险"
        st.metric("示例订单延迟概率", f"{probability * 100:.1f}%", level)
        st.dataframe(sample, use_container_width=True, hide_index=True)
    except ValueError:
        pass


def assistant_page(mart: pd.DataFrame):
    st.subheader("AI 分析助手")
    st.info("助手采用‘意图识别 → 受控分析函数 → DuckDB/指标计算 → AI 解释’流程。没有 API Key 时自动使用本地模板，不影响演示。")
    question = st.text_input("请输入业务问题", value="哪些地区的订单延迟率最高？")
    if st.button("开始分析", type="primary"):
        answer = answer_question(mart, question)
        if not answer["validation"]["ok"]:
            st.warning(answer["validation"]["reason"])
            return
        table = answer["table"]
        st.markdown("#### 结构化意图")
        st.json(answer["intent"])
        st.markdown("#### 数据结果")
        st.dataframe(table, use_container_width=True, hide_index=True)
        if not table.empty and len(table.columns) >= 2:
            numeric = [column for column in table.columns if pd.api.types.is_numeric_dtype(table[column])]
            if numeric:
                fig = px.bar(table, x=table.columns[0], y=numeric[0], title="分析结果可视化")
                st.plotly_chart(fig, use_container_width=True)
        result = table.to_dict(orient="records")
        st.markdown("#### AI 分析结论")
        st.markdown(generate_insight(question, result, answer["metric_name"]))


def abtest_page():
    st.subheader("A/B 测试计算器")
    st.caption("Olist 数据不是随机实验数据，本页面使用独立的实验参数演示统计检验，不将观察性分析误写成 A/B 测试。")
    left, right = st.columns(2)
    with left:
        conversions_a = st.number_input("A 组转化数", min_value=0, value=120)
        visitors_a = st.number_input("A 组样本数", min_value=1, value=1000)
    with right:
        conversions_b = st.number_input("B 组转化数", min_value=0, value=155)
        visitors_b = st.number_input("B 组样本数", min_value=1, value=1000)
    result = proportion_test(conversions_a, visitors_a, conversions_b, visitors_b)
    cols = st.columns(4)
    cols[0].metric("A 组转化率", pct(result["rate_a"]))
    cols[1].metric("B 组转化率", pct(result["rate_b"]))
    cols[2].metric("相对提升", f"{result['lift_pct']:.1f}%")
    cols[3].metric("p-value", f"{result['p_value']:.4f}")
    st.success(result["conclusion"] if result["significant"] else result["conclusion"])


def main():
    db_path = ensure_warehouse()
    mart = load_data(db_path)
    st.title("📈 Olist 电商经营分析 Copilot")
    st.caption("数据分析 + 机器学习 + 受控式 AI 辅助决策平台")
    filtered = render_sidebar(mart)
    tabs = st.tabs(["经营总览", "商品与卖家", "履约与预测", "AI 分析助手", "A/B 测试"])
    with tabs[0]:
        overview_page(filtered)
    with tabs[1]:
        product_seller_page(filtered)
    with tabs[2]:
        delivery_page(filtered)
    with tabs[3]:
        assistant_page(filtered)
    with tabs[4]:
        abtest_page()
    st.divider()
    st.caption("项目说明：指标口径集中维护；AI 不直接执行自由 SQL；模型只使用预测时点可获得的特征。")


if __name__ == "__main__":
    main()
