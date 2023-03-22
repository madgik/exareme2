import pytest

from mipengine.node.monetdb_interface.monet_db_facade import make_idempotent
from mipengine.node.monetdb_interface.monet_db_facade import (
    make_udf_execution_idempotent,
)


@pytest.mark.parametrize(
    "query, expected_idempotent_query",
    [
        (
            "CREATE TABLE my_table (id INT, name VARCHAR(255))",
            "CREATE TABLE IF NOT EXISTS my_table (id INT, name VARCHAR(255))",
        ),
        (
            "CREATE REMOTE TABLE my_table (id INT, name VARCHAR(255))",
            "CREATE REMOTE TABLE IF NOT EXISTS my_table (id INT, name VARCHAR(255))",
        ),
        (
            "CREATE VIEW my_view AS SELECT * FROM my_table",
            "CREATE OR REPLACE VIEW my_view AS SELECT * FROM my_table",
        ),
        ("SELECT * FROM my_table WHERE id = 1", "SELECT * FROM my_table WHERE id = 1"),
    ],
)
def test_make_idempotent(query, expected_idempotent_query):
    assert make_idempotent(query) == expected_idempotent_query


def test_make_udf_execution_idempotent():
    assert (
        make_udf_execution_idempotent("INSERT INTO my_tbl1 VALUES (1);")
        == "INSERT INTO my_tbl1 VALUES (1)\n"
        "WHERE NOT EXISTS (SELECT * FROM my_tbl1);"
    )
