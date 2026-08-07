from pathlib import Path

from olist_copilot.warehouse.connection import connect_readonly


def test_readonly_connection_can_query_built_warehouse(tmp_path: Path):
    import duckdb

    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE demo AS SELECT 1 AS value")
    con.close()

    readonly = connect_readonly(db_path)
    assert readonly.execute("SELECT value FROM demo").fetchone()[0] == 1
    readonly.close()
