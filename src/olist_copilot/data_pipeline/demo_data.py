"""Generate a deterministic, Olist-shaped demo dataset.

The project runs without external data by using this dataset. Real Olist CSV files
can replace the generated files in data/raw/ without changing the analytics code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


REQUIRED_TABLES = (
    "orders",
    "order_items",
    "payments",
    "reviews",
    "products",
    "sellers",
    "customers",
)


def generate_demo_dataset(output_dir: Path, seed: int = 42, n_orders: int = 240) -> Dict[str, Path]:
    """Create a small reproducible Olist-like dataset and return table paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    states = np.array(["SP", "RJ", "MG", "BA", "PR", "RS"])
    cities = {
        "SP": "Sao Paulo",
        "RJ": "Rio de Janeiro",
        "MG": "Belo Horizonte",
        "BA": "Salvador",
        "PR": "Curitiba",
        "RS": "Porto Alegre",
    }
    categories = np.array(["health_beauty", "bed_bath_table", "computers_accessories", "sports_leisure", "watches_gifts", "toys"])

    product_ids = np.array([f"prod_{i:03d}" for i in range(36)])
    products = pd.DataFrame(
        {
            "product_id": product_ids,
            "product_category_name": rng.choice(categories, size=len(product_ids)),
            "product_weight_g": rng.integers(200, 6500, size=len(product_ids)),
            "product_length_cm": rng.integers(10, 70, size=len(product_ids)),
            "product_height_cm": rng.integers(5, 45, size=len(product_ids)),
            "product_width_cm": rng.integers(8, 55, size=len(product_ids)),
        }
    )

    seller_ids = np.array([f"seller_{i:03d}" for i in range(18)])
    seller_states = rng.choice(states, size=len(seller_ids), p=[0.42, 0.16, 0.14, 0.1, 0.1, 0.08])
    sellers = pd.DataFrame(
        {
            "seller_id": seller_ids,
            "seller_zip_code_prefix": rng.integers(1000, 99999, size=len(seller_ids)),
            "seller_city": [cities[s] for s in seller_states],
            "seller_state": seller_states,
        }
    )

    customer_ids = np.array([f"customer_{i:04d}" for i in range(max(50, n_orders // 2))])
    customer_states = rng.choice(states, size=len(customer_ids), p=[0.44, 0.17, 0.13, 0.1, 0.09, 0.07])
    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "customer_unique_id": [f"unique_{i:04d}" for i in range(len(customer_ids))],
            "customer_zip_code_prefix": rng.integers(1000, 99999, size=len(customer_ids)),
            "customer_city": [cities[s] for s in customer_states],
            "customer_state": customer_states,
        }
    )

    order_ids = np.array([f"order_{i:05d}" for i in range(n_orders)])
    customer_choices = rng.choice(customer_ids, size=n_orders)
    purchase_dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, size=n_orders), unit="D")
    item_count = rng.integers(1, 4, size=n_orders)
    seller_choices = rng.choice(seller_ids, size=n_orders)
    product_choices = rng.choice(product_ids, size=n_orders)
    customer_state_lookup = dict(zip(customers["customer_id"], customers["customer_state"]))
    seller_state_lookup = dict(zip(sellers["seller_id"], sellers["seller_state"]))
    cross_state = np.array([customer_state_lookup[c] != seller_state_lookup[s] for c, s in zip(customer_choices, seller_choices)])
    estimated_days = np.where(cross_state, rng.integers(12, 24, size=n_orders), rng.integers(5, 15, size=n_orders))
    actual_days = np.where(cross_state, rng.integers(8, 29, size=n_orders), rng.integers(3, 19, size=n_orders))
    delivered = purchase_dates + pd.to_timedelta(actual_days, unit="D")
    estimated = purchase_dates + pd.to_timedelta(estimated_days, unit="D")
    approved = purchase_dates + pd.to_timedelta(rng.integers(0, 2, size=n_orders), unit="D")
    carrier = purchase_dates + pd.to_timedelta(rng.integers(1, 5, size=n_orders), unit="D")
    statuses = np.where(rng.random(n_orders) < 0.96, "delivered", "canceled")
    delivered = pd.Series(delivered).where(statuses == "delivered")
    late_flag = (delivered.notna() & (delivered > estimated)).astype(float).where(delivered.notna())

    orders = pd.DataFrame(
        {
            "order_id": order_ids,
            "customer_id": customer_choices,
            "order_status": statuses,
            "order_purchase_timestamp": purchase_dates,
            "order_approved_at": approved,
            "order_delivered_carrier_date": carrier,
            "order_delivered_customer_date": delivered,
            "order_estimated_delivery_date": estimated,
            "late_flag": late_flag,
        }
    )

    item_rows = []
    payment_rows = []
    review_rows = []
    for i, order_id in enumerate(order_ids):
        count = int(item_count[i])
        for item_number in range(count):
            product_id = product_choices[i] if item_number == 0 else rng.choice(product_ids)
            seller_id = seller_choices[i] if item_number == 0 else rng.choice(seller_ids)
            price = round(float(rng.uniform(18, 420)), 2)
            freight = round(float(rng.uniform(5, 48) + (12 if cross_state[i] else 0)), 2)
            item_rows.append(
                {
                    "order_id": order_id,
                    "order_item_id": item_number + 1,
                    "product_id": product_id,
                    "seller_id": seller_id,
                    "shipping_limit_date": purchase_dates[i] + pd.Timedelta(days=3),
                    "price": price,
                    "freight_value": freight,
                }
            )
        total = sum(row["price"] + row["freight_value"] for row in item_rows if row["order_id"] == order_id)
        payment_rows.append(
            {
                "order_id": order_id,
                "payment_sequential": 1,
                "payment_type": rng.choice(["credit_card", "boleto", "voucher", "debit_card"], p=[0.62, 0.2, 0.1, 0.08]),
                "payment_installments": int(rng.integers(1, 8)),
                "payment_value": round(total, 2),
            }
        )
        score = int(np.clip(round(4.5 - 1.4 * float(late_flag.iloc[i] or 0) + rng.normal(0, 0.8)), 1, 5)) if statuses[i] == "delivered" else np.nan
        review_rows.append({"review_id": f"review_{i:05d}", "order_id": order_id, "review_score": score})

    order_items = pd.DataFrame(item_rows)
    payments = pd.DataFrame(payment_rows)
    reviews = pd.DataFrame(review_rows)

    tables = {
        "orders": orders,
        "order_items": order_items,
        "payments": payments,
        "reviews": reviews,
        "products": products,
        "sellers": sellers,
        "customers": customers,
    }
    paths: Dict[str, Path] = {}
    filename_map = {
        "orders": "olist_orders_dataset.csv",
        "order_items": "olist_order_items_dataset.csv",
        "payments": "olist_order_payments_dataset.csv",
        "reviews": "olist_order_reviews_dataset.csv",
        "products": "olist_products_dataset.csv",
        "sellers": "olist_sellers_dataset.csv",
        "customers": "olist_customers_dataset.csv",
    }
    for name, frame in tables.items():
        path = output_dir / filename_map[name]
        frame.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")
        paths[name] = path
    translation = pd.DataFrame({"product_category_name": categories, "product_category_name_english": categories})
    translation_path = output_dir / "product_category_name_translation.csv"
    translation.to_csv(translation_path, index=False)
    paths["translation"] = translation_path
    return paths
