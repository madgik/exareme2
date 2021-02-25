import json

import pymonetdb
import pytest

from mipengine.node.monetdb_interface.common import cursor, connection
from mipengine.node.tasks.data_classes import ColumnInfo
from mipengine.node.tasks.data_classes import TableSchema
from mipengine.tests.node.set_up_nodes import celery_local_node_1
from mipengine.utils.custom_exception import IncompatibleSchemasMergeException, TableCannotBeFound

create_table = celery_local_node_1.signature('mipengine.node.tasks.tables.create_table')
create_merge_table = celery_local_node_1.signature('mipengine.node.tasks.merge_tables.create_merge_table')
get_merge_tables = celery_local_node_1.signature('mipengine.node.tasks.merge_tables.get_merge_tables')
clean_up = celery_local_node_1.signature('mipengine.node.tasks.common.clean_up')

context_id = "regrEssion"


def setup_tables_for_merge(number_of_table: int) -> str:
    schema = TableSchema([ColumnInfo("col1", "INT"), ColumnInfo("col2", "FLOAT"), ColumnInfo("col3", "TEXT")])
    json_schema = schema.to_json()
    table_name = create_table.delay(f"{context_id}_table{number_of_table}",
                                    str(pymonetdb.uuid.uuid1()).replace("-", ""), json_schema).get()
    cursor.execute(
        f"INSERT INTO {table_name} VALUES ( {number_of_table}, {number_of_table}, 'table_{number_of_table}')")
    connection.commit()
    return table_name


def test_incompatible_schemas_merge():
    with pytest.raises(IncompatibleSchemasMergeException):
        incompatible_table = setup_tables_for_merge(5)
        cursor.execute(f"ALTER TABLE {incompatible_table} DROP {'col1'}")
        connection.commit()

        incompatible_partition_tables = [setup_tables_for_merge(1), setup_tables_for_merge(2),
                                         setup_tables_for_merge(3), setup_tables_for_merge(4), incompatible_table]
        create_merge_table.delay(context_id, str(pymonetdb.uuid.uuid1()).replace("-", ""),
                                 json.dumps(incompatible_partition_tables)).get()


def test_table_cannot_be_found():
    with pytest.raises(TableCannotBeFound):
        not_found_tables = [setup_tables_for_merge(1), setup_tables_for_merge(2), setup_tables_for_merge(3),
                            setup_tables_for_merge(4), "non_existant_table"]

        create_merge_table.delay(context_id, str(pymonetdb.uuid.uuid1()).replace("-", ""),
                                 json.dumps(not_found_tables)).get()


def test_sql_injection_get_merge_tables():
    with pytest.raises(ValueError):
        get_merge_tables.delay("drop table data;").get()


def test_sql_injection_create_merge_table_context_id():
    with pytest.raises(ValueError):
        success_partition_tables = [setup_tables_for_merge(1), setup_tables_for_merge(2), setup_tables_for_merge(3),
                                    setup_tables_for_merge(4)]
        create_merge_table.delay("drop table data;",
                                 str(pymonetdb.uuid.uuid1()).replace("-", ""),
                                 json.dumps(success_partition_tables)).get()


def test_sql_injection_create_merge_table_uuid():
    with pytest.raises(ValueError):
        success_partition_tables = [setup_tables_for_merge(1), setup_tables_for_merge(2), setup_tables_for_merge(3),
                                    setup_tables_for_merge(4)]
        create_merge_table.delay(context_id, "drop table data;",
                                 json.dumps(success_partition_tables)).get()


def test_sql_injection_create_merge_table_table_names():
    with pytest.raises(ValueError):
        create_merge_table.delay(context_id, str(pymonetdb.uuid.uuid1()).replace("-", ""),
                                 json.dumps(["drop table data;"])).get()


def test_merge_tables():
    success_partition_tables = [setup_tables_for_merge(1), setup_tables_for_merge(2), setup_tables_for_merge(3),
                                setup_tables_for_merge(4)]
    success_merge_table_1_name = create_merge_table.delay(context_id, str(pymonetdb.uuid.uuid1()).replace("-", ""),
                                                          json.dumps(success_partition_tables)).get()
    merge_tables = get_merge_tables.delay(context_id).get()
    assert success_merge_table_1_name in merge_tables

    clean_up.delay(context_id.lower()).get()


test_merge_tables()