from olist_copilot.config import DATA_ROOT, DEFAULT_DB_PATH
from olist_copilot.data_pipeline.pipeline import build_warehouse, discover_olist_files


if __name__ == "__main__":
    files = discover_olist_files(DATA_ROOT / "raw")
    summary = build_warehouse(files, DEFAULT_DB_PATH)
    print(f"Built {DEFAULT_DB_PATH}")
    for name, count in summary.items():
        print(f"{name}: {count}")
