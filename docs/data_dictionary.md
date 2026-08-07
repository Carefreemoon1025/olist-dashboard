# 数据字典与口径说明

## 订单延迟定义

订单只有在以下条件同时满足时才参与延迟率计算：

1. `order_status = delivered`；
2. 存在实际送达时间；
3. 存在预计送达时间。

若实际送达时间晚于预计送达时间，则 `late_flag = 1`，否则为 `0`。

## 机器学习标签

`late_flag` 是预测目标，不作为输入特征。
模型特征必须来自订单早期可获得的信息，因此不使用实际送达时间、评价分数和最终履约状态。

## 有效订单

`canceled` 和 `unavailable` 订单不纳入经营总览的订单量、支付金额和客单价计算。

## 主要分析表

### `mart_order_analysis`

| 字段 | 说明 |
|---|---|
| `order_id` | 订单 ID |
| `customer_state` | 用户所在州 |
| `seller_state` | 卖家所在州 |
| `product_category` | 主要商品品类 |
| `order_total_value` | 商品金额 |
| `freight_value` | 运费 |
| `item_count` | 订单商品数量 |
| `product_weight_g` | 商品平均重量 |
| `order_purchase_timestamp` | 下单时间 |
| `delivery_days` | 实际配送天数 |
| `estimated_days` | 预计配送天数 |
| `late_flag` | 延迟标签 |
| `review_score` | 评价分数 |
