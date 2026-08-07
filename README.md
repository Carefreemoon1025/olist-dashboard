# Olist 电商经营分析 Copilot

> 面向数据分析 / AI 应用岗位的可复现求职项目：用 SQL、Python、BI 可视化、机器学习和受控式大模型分析完成一条电商经营分析闭环。

## 项目定位

本项目不是一个“调用大模型生成报告”的简单 Demo，而是一个模块化的数据分析应用：

```text
Olist CSV
  → Pandas 数据清洗
  → DuckDB 分析仓库
  → 统一指标语义层
  → Streamlit 经营看板
  → 订单延迟风险模型
  → 受控式自然语言分析助手
  → AI 证据型分析报告
```

## 主要功能

| 模块 | 求职能力证据 |
|---|---|
| 数据接入与清洗 | Pandas、多表关联、数据质量处理 |
| DuckDB 分析仓库 | SQL、事实表/分析宽表、可复现 ETL |
| 指标语义层 | 指标口径设计、数据仓库思维 |
| 经营分析看板 | Streamlit、Plotly、BI 交互和可视化 |
| 延迟风险预测 | Logistic Regression、分类评估、防止数据泄漏 |
| AI 分析助手 | 意图识别、工具调用、参数化查询、LLM 解释 |
| A/B 测试 | SciPy、比例检验、显著性判断 |

## 技术栈

- Python 3.10+
- Pandas / NumPy
- DuckDB
- Streamlit
- Plotly
- Scikit-learn
- SciPy
- OpenAI-compatible API（可接入 DeepSeek、智谱、通义等兼容接口）
- Pytest

## 快速开始

### 1. 创建环境并安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

### 2. 生成演示数据并构建仓库

项目已经包含小规模、固定随机种子的演示数据。需要重新生成时运行：

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts\generate_demo_data.py
python scripts\build_warehouse.py
```

### 3. 启动应用

```powershell
$env:PYTHONPATH = "$PWD\src"
streamlit run app.py
```

也可以使用：

```powershell
.\run.ps1
```

默认数据存放在：

```text
data/raw/
data/warehouse/olist.duckdb
```

## 使用真实 Olist 数据

将真实数据按 Kaggle Olist 的标准文件名放入 `data/raw/`：

```text
olist_orders_dataset.csv
olist_order_items_dataset.csv
olist_order_payments_dataset.csv
olist_order_reviews_dataset.csv
olist_products_dataset.csv
olist_sellers_dataset.csv
olist_customers_dataset.csv
```

然后重新构建：

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts\build_warehouse.py
streamlit run app.py
```

项目会优先读取完整的真实数据；如果数据不完整，应用会使用可复现的演示数据。

## 可选：启用大模型分析

没有 API Key 时，AI 分析助手使用本地模板，项目仍然可以完整运行。需要接入兼容 OpenAI SDK 的模型时，设置：

```powershell
$env:LLM_API_KEY = "your-api-key"
$env:LLM_BASE_URL = "https://your-compatible-endpoint/v1"
$env:LLM_MODEL = "your-model-name"
```

大模型的职责只包括：

1. 识别用户问题的分析意图；
2. 调用受控的分析函数；
3. 基于真实计算结果生成解释。

它不会直接执行任意 SQL，也不会替代数据库完成核心计算。

## 项目结构

```text
olist-analysis-copilot/
├── app.py
├── pyproject.toml
├── requirements.txt
├── run.ps1
├── config/
├── data/
│   ├── raw/
│   ├── processed/
│   └── warehouse/
├── src/olist_copilot/
│   ├── data_pipeline/       # 数据读取、清洗、演示数据、建仓
│   ├── warehouse/           # 数据库连接与查询
│   ├── metrics/             # 统一指标定义与计算
│   ├── analytics/           # 看板和 AI 共用的分析服务
│   ├── ml/                  # 延迟风险预测
│   └── ai/                  # 意图识别、护栏、查询编排、LLM
├── tests/                   # 自动化测试
├── docs/                    # 数据字典、指标口径、架构和模型报告
├── models/                  # 训练后模型产物
└── scripts/                 # 数据初始化和建仓脚本
```

## 可讲述的工程设计

### 1. 为什么使用 DuckDB

项目是本地分析型应用，DuckDB 零配置、支持 SQL 和列式分析，并且可以直接与 Pandas 互操作。相比为了演示而引入 MySQL、Redis 和微服务，DuckDB 更符合在校生项目的可复现性。

### 2. 为什么不让大模型直接生成 SQL

项目采用：

```text
自然语言
  → 结构化意图
  → 指标和维度白名单
  → 参数化分析函数
  → DuckDB 查询
  → AI 解释
```

这样可以统一指标口径、降低 SQL 幻觉风险，并且让结果可测试、可追溯。

### 3. 如何避免模型数据泄漏

延迟预测只使用下单或支付确认时已经可以获得的特征，例如商品价格、运费、地区、商品数量和预计配送时间，不使用实际送达时间和评价分数。

## 测试

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m pytest -q -p no:cacheprovider
```

测试覆盖：

- 演示数据可重复生成；
- DuckDB 仓库构建；
- KPI 口径；
- 延迟风险特征防泄漏；
- 模型训练和评估；
- AI 意图识别和安全护栏。

## 面试演示顺序

1. 经营总览：展示指标体系和趋势；
2. 履约分析：说明延迟率与评价的关系；
3. 风险模型：展示评估指标和防数据泄漏设计；
4. AI 助手：输入“哪些地区的订单延迟率最高？”；
5. 说明 AI 只负责交互和解释，核心计算由 DuckDB 和指标服务完成。

## 简历项目描述示例

> **Olist 电商经营分析 Copilot**：基于 Olist 电商订单数据，使用 Pandas、DuckDB 和 Streamlit 构建订单、商品、卖家及地区指标体系，实现经营表现、物流履约和用户评价的交互式分析；使用 Logistic Regression 建立订单延迟风险预测模型，并通过受控式自然语言查询和 OpenAI-compatible LLM 自动生成带数据依据的经营分析报告。
