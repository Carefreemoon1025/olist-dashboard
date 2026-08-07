from pathlib import Path

from olist_copilot.config import DATA_ROOT
from olist_copilot.data_pipeline.demo_data import generate_demo_dataset


if __name__ == "__main__":
    output = DATA_ROOT / "raw"
    paths = generate_demo_dataset(output, seed=42, n_orders=240)
    for name, path in paths.items():
        print(f"{name}: {path}")
