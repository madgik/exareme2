import pymonetdb
import pytest

from mipengine.common.node_tasks_DTOs import TableData
from mipengine.tests.node import nodes_communication

local_node_id = "localnode1"
local_node = nodes_communication.get_celery_app(local_node_id)
local_node_create_view = nodes_communication.get_celery_create_view_signature(local_node)
local_node_get_views = nodes_communication.get_celery_get_views_signature(local_node)
local_node_get_view_data = nodes_communication.get_celery_get_table_data_signature(local_node)
local_node_cleanup = nodes_communication.get_celery_cleanup_signature(local_node)

context_id = "filters"


@pytest.fixture(autouse=True)
def cleanup_views():
    yield

    local_node_cleanup.delay(context_id=context_id.lower()).get()


def test_equal():
    columns = ["dataset", "age_value", "gcs_motor_response_scale", "pupil_reactivity_right_eye_result"]
    datasets = ["dummy_tbi"]
    pathology = "tbi"
    filters = {
        "condition": "AND",
        "rules": [
            {
                "id": "age_value",
                "field": "age_value",
                "type": "int",
                "input": "number",
                "operator": "equal",
                "value": 17
            }
        ],
        "valid": True
    }

    view_name = local_node_create_view.delay(context_id=context_id,
                                             command_id=str(pymonetdb.uuid.uuid1()).replace("-", ""),
                                             pathology=pathology,
                                             datasets=datasets,
                                             columns=columns,
                                             filters=filters
                                             ).get()
    views = local_node_get_views.delay(context_id=context_id).get()
    assert view_name in views

    view_data_json = local_node_get_view_data.delay(table_name=view_name).get()
    view_data = TableData.from_json(view_data_json)
    for row in view_data.data:
        assert row[0] == "dummy_tbi"
        assert row[1] == 17


def test_equal_or_not_equal():
    columns = ["dataset", "age_value", "gcs_motor_response_scale", "pupil_reactivity_right_eye_result"]
    datasets = ["dummy_tbi"]
    pathology = "tbi"
    filters = {
        "condition": "OR",
        "rules": [
            {
                "id": "age_value",
                "field": "age_value",
                "type": "int",
                "input": "number",
                "operator": "equal",
                "value": 17
            },
            {
                "id": "pupil_reactivity_right_eye_result",
                "field": "pupil_reactivity_right_eye_result",
                "type": "string",
                "input": "text",
                "operator": "not_equal",
                "value": "Nonreactive"
            }
        ],
        "valid": True
    }

    view_name = local_node_create_view.delay(context_id=context_id,
                                             command_id=str(pymonetdb.uuid.uuid1()).replace("-", ""),
                                             pathology=pathology,
                                             datasets=datasets,
                                             columns=columns,
                                             filters=filters
                                             ).get()
    views = local_node_get_views.delay(context_id=context_id).get()
    assert view_name in views

    view_data_json = local_node_get_view_data.delay(table_name=view_name).get()
    view_data = TableData.from_json(view_data_json)
    for row in view_data.data:
        assert row[0] == "dummy_tbi"
        assert row[1] == 17 or row[3] != "Nonreactive"


def test_less_and_greater():
    columns = ["dataset", "age_value", "gcs_motor_response_scale", "pupil_reactivity_right_eye_result"]
    datasets = ["dummy_tbi"]
    pathology = "tbi"
    filters = {
        "condition": "AND",
        "rules": [
            {
                "id": "age_value",
                "field": "age_value",
                "type": "int",
                "input": "number",
                "operator": "less",
                "value": 50
            },
            {
                "id": "age_value",
                "field": "age_value",
                "type": "int",
                "input": "number",
                "operator": "greater",
                "value": 20
            }
        ],
        "valid": True
    }

    view_name = local_node_create_view.delay(context_id=context_id,
                                             command_id=str(pymonetdb.uuid.uuid1()).replace("-", ""),
                                             pathology=pathology,
                                             datasets=datasets,
                                             columns=columns,
                                             filters=filters
                                             ).get()
    views = local_node_get_views.delay(context_id=context_id).get()
    assert view_name in views

    view_data_json = local_node_get_view_data.delay(table_name=view_name).get()
    view_data = TableData.from_json(view_data_json)
    for row in view_data.data:
        assert row[0] == "dummy_tbi"
        assert row[1] < 50
        assert row[1] > 20


def test_between_and_not_between():
    columns = ["dataset", "age_value", "gcs_motor_response_scale", "mortality_core"]
    datasets = ["dummy_tbi"]
    pathology = "tbi"
    filters = {
        "condition": "AND",
        "rules": [
            {
                "id": "age_value",
                "field": "age_value",
                "type": "int",
                "input": "number",
                "operator": "not_between",
                "value": [
                    60,
                    90
                ]
            },
            {
                "id": "mortality_core",
                "field": "mortality_core",
                "type": "double",
                "input": "number",
                "operator": "between",
                "value": [
                    0.3,
                    0.8
                ]
            }
        ],
        "valid": True
    }

    view_name = local_node_create_view.delay(context_id=context_id,
                                             command_id=str(pymonetdb.uuid.uuid1()).replace("-", ""),
                                             pathology=pathology,
                                             datasets=datasets,
                                             columns=columns,
                                             filters=filters
                                             ).get()
    views = local_node_get_views.delay(context_id=context_id).get()
    assert view_name in views

    view_data_json = local_node_get_view_data.delay(table_name=view_name).get()
    view_data = TableData.from_json(view_data_json)
    for row in view_data.data:
        assert row[0] == "dummy_tbi"
        assert row[1] < 60 or row[1] > 90
        assert 0.3 <= row[3] <= 0.8


def test_is_null_or_is_not_null():
    columns = ["dataset", "gose_score", "gcs_total_score"]
    datasets = ["dummy_tbi"]
    pathology = "tbi"
    filters = {
        "condition": "OR",
        "rules": [
            {
                "id": "gose_score",
                "field": "gose_score",
                "type": "text",
                "input": "text",
                "operator": "is_null",
                "value": None
            },
            {
                "id": "gcs_total_score",
                "field": "gcs_total_score",
                "type": "int",
                "input": "number",
                "operator": "is_not_null",
                "value": None
            }
        ],
        "valid": True
    }

    view_name = local_node_create_view.delay(context_id=context_id,
                                             command_id=str(pymonetdb.uuid.uuid1()).replace("-", ""),
                                             pathology=pathology,
                                             datasets=datasets,
                                             columns=columns,
                                             filters=filters
                                             ).get()
    views = local_node_get_views.delay(context_id=context_id).get()
    assert view_name in views

    view_data_json = local_node_get_view_data.delay(table_name=view_name).get()
    view_data = TableData.from_json(view_data_json)
    assert view_data.data != []
    for row in view_data.data:
        assert row[0] == "dummy_tbi"
        assert row[1] == "null" or row[2] != "null"


def test_all():
    # View for females that of age 20-30 and gose_score is not null
    # or males with mortality_core greater than 0.5 and less or equal to 0.8
    columns = ["dataset",
               "age_value",
               "gender_type",
               "gose_score",
               "mortality_core",
               "mortality_gose"]
    datasets = ["dummy_tbi"]
    pathology = "tbi"
    filters = {
        "condition": "OR",
        "rules": [
            {
                "condition": "AND",
                "rules": [
                    {
                        "id": "gender_type",
                        "field": "gender_type",
                        "type": "string",
                        "input": "text",
                        "operator": "equal",
                        "value": "F"
                    },
                    {
                        "condition": "AND",
                        "rules": [
                            {
                                "id": "age_value",
                                "field": "age_value",
                                "type": "int",
                                "input": "number",
                                "operator": "between",
                                "value": [
                                    20,
                                    30
                                ]
                            },
                            {
                                "id": "gose_score",
                                "field": "gose_score",
                                "type": "text",
                                "input": "text",
                                "operator": "is_not_null",
                                "value": None
                            }
                        ]
                    }
                ]
            },
            {
                "condition": "AND",
                "rules": [
                    {
                        "id": "gender_type",
                        "field": "gender_type",
                        "type": "string",
                        "input": "text",
                        "operator": "not_equal",
                        "value": "F"
                    },
                    {
                        "condition": "AND",
                        "rules": [
                            {
                                "id": "mortality_core",
                                "field": "mortality_core",
                                "type": "double",
                                "input": "number",
                                "operator": "greater",
                                "value": 0.5
                            },
                            {
                                "id": "mortality_core",
                                "field": "mortality_core",
                                "type": "double",
                                "input": "number",
                                "operator": "less_or_equal",
                                "value": 0.8
                            }
                        ]
                    }
                ]
            }
        ],
        "valid": True
    }

    view_name = local_node_create_view.delay(context_id=context_id,
                                             command_id=str(pymonetdb.uuid.uuid1()).replace("-", ""),
                                             pathology=pathology,
                                             datasets=datasets,
                                             columns=columns,
                                             filters=filters
                                             ).get()
    views = local_node_get_views.delay(context_id=context_id).get()
    assert view_name in views

    view_data_json = local_node_get_view_data.delay(table_name=view_name).get()
    view_data = TableData.from_json(view_data_json)
    assert view_data.data != []
    for row in view_data.data:
        assert row[0] == "dummy_tbi"
        assert (row[2] == "F" and (20 <= row[1] <= 30 and row[3] != "null")) or (
                    row[2] != "F" and (row[4] > 0.5 or row[4] <= 0.8))
