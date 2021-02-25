import json

import pymonetdb
import pytest

from mipengine.node.tasks.data_classes import ColumnInfo
from mipengine.node.tasks.data_classes import TableData
from mipengine.node.tasks.data_classes import TableSchema
from mipengine.tests.node import nodes_communication

local_node_id = "local_node_1"
local_node = nodes_communication.get_celery_app(local_node_id)
local_node_create_table = nodes_communication.get_celery_create_table_signature(local_node)
local_node_create_view = nodes_communication.get_celery_create_view_signature(local_node)
local_node_get_views = nodes_communication.get_celery_get_views_signature(local_node)
local_node_get_view_data = nodes_communication.get_celery_get_view_data_signature(local_node)
local_node_get_view_schema = nodes_communication.get_celery_get_view_schema_signature(local_node)
local_node_cleanup = nodes_communication.get_celery_cleanup_signature(local_node)

context_id = "regrEssion"


@pytest.fixture(autouse=True)
def cleanup_tables():
    yield

    local_node_cleanup.delay(context_id.lower()).get()


def test_create_and_get_view():
    columns = ["dataset", "age_value", "gcs_motor_response_scale", "pupil_reactivity_right_eye_result"]
    datasets = ["edsd"]
    pathology = "tbi"
    view_name = local_node_create_view.delay(context_id,
                                             str(pymonetdb.uuid.uuid1()).replace("-", ""),
                                             pathology,
                                             json.dumps(datasets),
                                             json.dumps(columns),
                                             "filter"
                                             ).get()
    views = local_node_get_views.delay(context_id).get()
    assert view_name in views

    schema = TableSchema([
        ColumnInfo('dataset', 'text'),
        ColumnInfo('age_value', 'int'),
        ColumnInfo('gcs_motor_response_scale', 'text'),
        ColumnInfo('pupil_reactivity_right_eye_result', 'text')])
    schema_result_json = local_node_get_view_schema.delay(view_name).get()
    schema_result = TableSchema.from_json(schema_result_json)
    assert schema_result == schema

    table_data_json = local_node_get_view_data.delay(view_name).get()
    table_data = TableData.from_json(table_data_json)
    assert table_data.data is not []
    assert table_data.schema == schema
