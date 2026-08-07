from olist_copilot.config import DATA_ROOT, DEMO_DATA_ROOT
from olist_copilot.data_pipeline.demo_data import generate_demo_dataset


if __name__ == "__main__":
    output = DEMO_DATA_ROOT
    paths = generate_demo_dataset(output, seed=42, n_orders=240)
    for name, path in paths.items():
        print(f"{name}: {path}")
