import pytest

from mipengine.controller.algorithm_executor_node_data_objects import TableName
from mipengine.node.monetdb_interface.common_actions import create_table_name
from mipengine.node_tasks_DTOs import TableType


def test_create_table_name():
    tablename = create_table_name(
        table_type=TableType.NORMAL,
        node_id="nodeid",
        context_id="contextid",
        command_id="commandid",
        command_subid="commandsubid",
    )

    tablename_obj = TableName(tablename)

    assert tablename_obj.table_type == TableType.NORMAL
    assert tablename_obj.node_id == "nodeid"
    assert tablename_obj.context_id == "contextid"
    assert tablename_obj.command_id == "commandid"
    assert tablename_obj.command_subid == "commandsubid"


def test_create_table_name_with_bad_table_type():
    with pytest.raises(TypeError) as exc:
        create_table_name(
            table_type="badtabletype",
            node_id="nodeid",
            context_id="contextid",
            command_id="commandid",
            command_subid="commandsubid",
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
    "table_type, node_id, context_id, command_id, command_subid",
    get_test_create_table_name_with_bad_parameters_cases(),
)
def test_create_table_with_bad_parameters(
    table_type, node_id, context_id, command_id, command_subid
):
    with pytest.raises(ValueError) as exc:
        create_table_name(
            table_type=table_type,
            node_id=node_id,
            context_id=context_id,
            command_id=command_id,
            command_subid=command_subid,
        )

    assert "is not alphanumeric" in str(exc)
