import pytest

from mipengine.node.monetdb_interface.monet_db_facade import create_idempotent_query


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
def test_create_idempotent_query(query, expected_idempotent_query):
    assert create_idempotent_query(query) == expected_idempotent_query
