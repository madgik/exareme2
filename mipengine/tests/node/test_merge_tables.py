import json

import pytest

from mipengine.node.monetdb_interface.common import cursor, connection
from mipengine.node.node import app
from mipengine.node.tasks.data_classes import ColumnInfo
from mipengine.utils.custom_exception import IncompatibleSchemasMergeException, TableCannotBeFound

create_table = app.signature('mipengine.node.tasks.tables.create_table')
create_merge_table = app.signature('mipengine.node.tasks.merge_tables.create_merge_table')
get_merge_tables = app.signature('mipengine.node.tasks.merge_tables.get_merge_tables')
clean_up = app.signature('mipengine.node.tasks.common.clean_up')

context_id = "regression"


def setup_tables_for_merge(number_of_table: int) -> str:
    schema = [ColumnInfo("col1", "INT"), ColumnInfo("col2", "FLOAT"), ColumnInfo("col3", "TEXT")]
    json_schema = ColumnInfo.schema().dumps(schema, many=True)
    table_name = create_table.delay(f"table{number_of_table}", json_schema).get()
    connection.commit()
    cursor.execute(
        f"INSERT INTO {table_name} VALUES ( {number_of_table}, {number_of_table}, 'table_{number_of_table}' )")
    connection.commit()
    return table_name


def test_merge_tables():
    success_partition_tables = [setup_tables_for_merge(1), setup_tables_for_merge(2), setup_tables_for_merge(3),
                                setup_tables_for_merge(4)]

    success_merge_table_1_name = create_merge_table.delay(context_id, json.dumps(success_partition_tables)).get()
    merge_tables = get_merge_tables.delay(context_id).get()
    assert success_merge_table_1_name in merge_tables

    clean_up.delay(context_id).get()
    test_incompatible_schemas_merge()
    test_table_cannot_be_found()


def test_incompatible_schemas_merge():
    with pytest.raises(IncompatibleSchemasMergeException):
        incompatible_table = setup_tables_for_merge(5)
        cursor.execute(f"ALTER TABLE {incompatible_table} DROP {'col1'};")
        connection.commit()

        incompatible_partition_tables = [setup_tables_for_merge(1), setup_tables_for_merge(2),
                                         setup_tables_for_merge(3), setup_tables_for_merge(4), incompatible_table]
        return create_merge_table.delay(context_id, json.dumps(incompatible_partition_tables)).get()


def test_table_cannot_be_found():
    with pytest.raises(TableCannotBeFound):
        not_found_tables = [setup_tables_for_merge(1), setup_tables_for_merge(2), setup_tables_for_merge(3),
                            setup_tables_for_merge(4), "non_existant_table"]

        return create_merge_table.delay(context_id, json.dumps(not_found_tables)).get()

test_merge_tables()