from olist_copilot.config import DEFAULT_DB_PATH, DEMO_DATA_ROOT, DEMO_DB_PATH, RAW_DATA_ROOT
from olist_copilot.data_pipeline.demo_data import generate_demo_dataset
from olist_copilot.data_pipeline.pipeline import REQUIRED_TABLES, build_warehouse, discover_olist_files, resolve_source_files, warehouse_is_fresh


if __name__ == "__main__":
    raw_files = discover_olist_files(RAW_DATA_ROOT)
    if not REQUIRED_TABLES.issubset(raw_files):
        demo_files = discover_olist_files(DEMO_DATA_ROOT)
        if not REQUIRED_TABLES.issubset(demo_files):
            generate_demo_dataset(DEMO_DATA_ROOT, seed=42, n_orders=240)
    files, is_demo = resolve_source_files(RAW_DATA_ROOT, DEMO_DATA_ROOT)
    db_path = DEMO_DB_PATH if is_demo else DEFAULT_DB_PATH
    if not warehouse_is_fresh(db_path, files):
        summary = build_warehouse(files, db_path)
        print(f"Built {db_path}")
        for name, count in summary.items():
            print(f"{name}: {count}")
    else:
        print(f"Warehouse is up to date: {db_path}")
