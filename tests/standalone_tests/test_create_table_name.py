import pytest

from exareme2.worker.exareme2.tables.tables_service import create_table_name
from exareme2.worker_communication import TableInfo
from exareme2.worker_communication import TableSchema
from exareme2.worker_communication import TableType


def test_create_table_name():
    tablename = create_table_name(
        table_type=TableType.NORMAL,
        node_id="nodeid",
        context_id="contextid",
        command_id="commandid",
        result_id="commandsubid",
    )

    tablename_obj = TableInfo(
        name=tablename, schema_=TableSchema(columns=[]), type_=TableType.NORMAL
    )

    assert tablename_obj.type_ == TableType.NORMAL
    assert tablename_obj.node_id == "nodeid"
    assert tablename_obj.context_id == "contextid"
    assert tablename_obj.command_id == "commandid"
    assert tablename_obj.result_id == "commandsubid"


def test_create_table_name_with_bad_table_type():
    with pytest.raises(TypeError) as exc:
        create_table_name(
            table_type="badtabletype",
            node_id="nodeid",
            context_id="contextid",
            command_id="commandid",
            result_id="commandsubid",
        )

    assert "Table type is not acceptable: " in str(exc)


def get_test_create_table_name_with_bad_parameters_cases():
    test_create_table_name_with_bad_parameters_cases = [
        (
            TableType.NORMAL,
            "nodeid_1",
            "contextid2",
            "commandid3",
            "commandsubid4",
        ),
        (
            TableType.NORMAL,
            "nodeid1",
            "contextid_2",
            "commandid3",
            "commandsubid4",
        ),
        (
            TableType.NORMAL,
            "nodeid1",
            "contextid2",
            "commandid_3",
            "commandsubid4",
        ),
        (
            TableType.NORMAL,
            "nodeid1",
            "contextid2",
            "commandid3",
            "commandsubid_4",
        ),
        (
            TableType.NORMAL,
            "nodeid1",
            "contextid2",
            "commandid3",
            "commandsubid4.",
        ),
        (
            TableType.NORMAL,
            "nodeid1",
            "contextid2",
            "commandid3",
            "commandsubid4!",
        ),
        (
            TableType.NORMAL,
            "nodeid1",
            "contextid2",
            "commandid3",
            "commandsubid4+",
        ),
    ]
    return test_create_table_name_with_bad_parameters_cases


@pytest.mark.parametrize(
    "table_type, node_id, context_id, command_id, result_id",
    get_test_create_table_name_with_bad_parameters_cases(),
)
def test_create_table_with_bad_parameters(
    table_type, node_id, context_id, command_id, result_id
):
    with pytest.raises(ValueError) as exc:
        create_table_name(
            table_type=table_type,
            node_id=node_id,
            context_id=context_id,
            command_id=command_id,
            result_id=result_id,
        )

    assert "is not alphanumeric" in str(exc)
