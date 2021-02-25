import json

import pymonetdb
import pytest

from mipengine.node.tasks.data_classes import ColumnInfo
from mipengine.node.tasks.data_classes import TableData
from mipengine.node.tasks.data_classes import TableSchema
from mipengine.tests.node.set_up_nodes import celery_local_node_1

create_table = celery_local_node_1.signature('mipengine.node.tasks.tables.create_table')
create_view = celery_local_node_1.signature('mipengine.node.tasks.views.create_view')
get_views = celery_local_node_1.signature('mipengine.node.tasks.views.get_views')
get_view_data = celery_local_node_1.signature('mipengine.node.tasks.views.get_view_data')
get_view_schema = celery_local_node_1.signature('mipengine.node.tasks.views.get_view_schema')
clean_up = celery_local_node_1.signature('mipengine.node.tasks.common.clean_up')


def test_views():
    context_id = "regrEssion"
    columns = ["dataset", "age_value", "gcs_motor_response_scale", "pupil_reactivity_right_eye_result"]
    datasets = ["edsd"]
    pathology = "tbi"
    table_1_name = create_view.delay(context_id, str(pymonetdb.uuid.uuid1()).replace("-", ""), pathology, json.dumps(datasets), json.dumps(columns), "dsa").get()
    tables = get_views.delay(context_id).get()
    assert table_1_name in tables
    schema = TableSchema([
        ColumnInfo('dataset', 'text'),
        ColumnInfo('age_value', 'int'),
        ColumnInfo('gcs_motor_response_scale', 'text'),
        ColumnInfo('pupil_reactivity_right_eye_result', 'text')])

    schema_result = get_view_schema.delay(table_1_name).get()
    object_schema_result = TableSchema.from_json(schema_result)
    assert object_schema_result == schema
    table_data_json = get_view_data.delay(table_1_name).get()
    table_data = TableData.from_json(table_data_json)
    assert table_data.data is not []
    assert table_data.schema == schema

    clean_up.delay(context_id.lower()).get()


def test_sql_injection_get_view_data():
    with pytest.raises(ValueError):
        get_view_data.delay("drop table data;").get()


def test_sql_injection_get_views():
    with pytest.raises(ValueError):
        get_views.delay("drop table data;").get()


def test_sql_injection_get_view_schema():
    with pytest.raises(ValueError):
        get_view_schema.delay("drop table data;").get()


def test_sql_injection_create_view_context_id():
    with pytest.raises(ValueError):
        context_id = "drop table data;"
        columns = ["dataset", "age_value", "gcs_motor_response_scale", "pupil_reactivity_right_eye_result"]
        datasets = ["edsd"]
        pathology = "tbi"
        create_view.delay(context_id, str(pymonetdb.uuid.uuid1()).replace("-", ""), pathology, json.dumps(datasets), json.dumps(columns), "").get()


def test_sql_injection_create_view_columns():
    with pytest.raises(ValueError):
        context_id = "regrEssion"
        columns = ["drop table data;", "age_value", "gcs_motor_response_scale", "pupil_reactivity_right_eye_result"]
        datasets = ["edsd"]
        pathology = "tbi"
        create_view.delay(context_id, str(pymonetdb.uuid.uuid1()).replace("-", ""), pathology, json.dumps(datasets), json.dumps(columns), "").get()


def test_sql_injection_create_view_datasets():
    with pytest.raises(ValueError):
        context_id = "regrEssion"
        columns = ["dataset", "age_value", "gcs_motor_response_scale", "pupil_reactivity_right_eye_result"]
        datasets = ["drop table data;"]
        pathology = "tbi"
        create_view.delay(context_id, str(pymonetdb.uuid.uuid1()).replace("-", ""), pathology, json.dumps(datasets), json.dumps(columns), "").get()


def test_sql_injection_create_view_uuid():
    with pytest.raises(ValueError):
        context_id = "regrEssion"
        columns = ["dataset", "age_value", "gcs_motor_response_scale", "pupil_reactivity_right_eye_result"]
        datasets = ["edsd"]
        pathology = "tbi"
        create_view.delay(context_id, str(pymonetdb.uuid.uuid1()).replace("-", ""), pathology, json.dumps(datasets), json.dumps(columns), "").get()


def test_sql_injection_create_view_pathology():
    with pytest.raises(ValueError):
        context_id = "regrEssion"
        columns = ["dataset", "age_value", "gcs_motor_response_scale", "pupil_reactivity_right_eye_result"]
        datasets = ["edsd"]
        pathology = "drop table data;"
        create_view.delay(context_id, str(pymonetdb.uuid.uuid1()).replace("-", ""), pathology, json.dumps(datasets), json.dumps(columns), "").get()
