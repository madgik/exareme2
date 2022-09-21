import pytest

from mipengine.node.monetdb_interface.common_actions import (
    _drop_table_by_type_and_context_id,
)
from mipengine.node.monetdb_interface.common_actions import _drop_udfs_by_context_id
from mipengine.node.monetdb_interface.common_actions import get_data_model_cdes
from mipengine.node.monetdb_interface.common_actions import (
    get_dataset_code_per_dataset_label,
)
from mipengine.node.monetdb_interface.common_actions import get_table_data
from mipengine.node.monetdb_interface.common_actions import get_table_names
from mipengine.node.monetdb_interface.common_actions import get_table_schema
from mipengine.node.monetdb_interface.common_actions import get_table_type
from mipengine.node.monetdb_interface.merge_tables import add_to_merge_table
from mipengine.node.monetdb_interface.merge_tables import create_merge_table
from mipengine.node.monetdb_interface.merge_tables import validate_tables_can_be_merged
from mipengine.node.monetdb_interface.remote_tables import create_remote_table
from mipengine.node.monetdb_interface.tables import create_table
from mipengine.node.monetdb_interface.views import create_view
from mipengine.node.sql_injection_guard import is_primary_data_table
from mipengine.node.sql_injection_guard import is_socket_address
from mipengine.node.sql_injection_guard import isalnum
from mipengine.node.sql_injection_guard import isalpha
from mipengine.node.sql_injection_guard import isdatamodel
from mipengine.node.sql_injection_guard import isidentifier
from mipengine.node.sql_injection_guard import sql_injection_guard


@sql_injection_guard("input", isalnum)
def dummy_method_with_one_input(input):
    return input


@sql_injection_guard("filter", isidentifier)
def dummy_method_with_filter_input(filter):
    return input


@sql_injection_guard("input", isalnum, optional=True)
def dummy_method_with_one_optional_input(input=None):
    return input


@sql_injection_guard("input1", isalnum)
@sql_injection_guard("input2", isalpha)
def dummy_method_with_two_inputs(input1, input2):
    return input1


def custom_validator(string):
    return string == "specific_string"


@sql_injection_guard("input1", isalnum)
@sql_injection_guard("input2", isalpha)
@sql_injection_guard("input3", custom_validator)
def dummy_method_with_custom_validator(input1, input2, input3):
    return input1


class DummyClass:
    pass


def get_parametrization_list_success_cases():
    return [
        pytest.param(
            dummy_method_with_one_input,
            ["alnum1"],
            {},
            id="param passed as posarg",
        ),
        pytest.param(
            dummy_method_with_one_input,
            [],
            {"input": "alnum1"},
            id="param passed as kwarg",
        ),
        pytest.param(
            dummy_method_with_two_inputs,
            ["alnum1", "alpha"],
            {},
            id="two params passed as posargs",
        ),
        pytest.param(
            dummy_method_with_two_inputs,
            [],
            {"input1": "alnum1", "input2": "alpha"},
            id="two params passed as kwargs",
        ),
        pytest.param(
            dummy_method_with_two_inputs,
            ["alnum1"],
            {"input2": "alpha"},
            id="two params passed as posarg and kwarg",
        ),
        pytest.param(
            dummy_method_with_custom_validator,
            ["alnum1", "alpha", "specific_string"],
            {},
            id="three params with custom validator",
        ),
        pytest.param(
            dummy_method_with_one_optional_input,
            [],
            {},
            id="optional input not provided",
        ),
        pytest.param(
            dummy_method_with_one_optional_input,
            ["alnum1"],
            {},
            id="optional input provided",
        ),
        pytest.param(
            dummy_method_with_one_input,
            [5],
            {},
            id="non string passed",
        ),
        pytest.param(
            dummy_method_with_one_input,
            [[5, 6, 7, "alnum1"]],
            {},
            id="list of ints passed",
        ),
        pytest.param(
            dummy_method_with_one_input,
            [{"x": "alnum1"}],
            {},
            id="dict passed",
        ),
        pytest.param(
            dummy_method_with_one_input,
            [DummyClass()],
            {},
            id="object is not being validated",
        ),
        pytest.param(
            dummy_method_with_one_input,
            [2],
            {},
            id="integer is not being validated",
        ),
        pytest.param(
            dummy_method_with_one_input,
            [["alnum1", "alnum2"]],
            {},
            id="str passed inside list",
        ),
        pytest.param(
            dummy_method_with_one_input,
            [{"alnum1": "alnum2"}],
            {},
            id="str passed inside dict",
        ),
        pytest.param(
            dummy_method_with_one_input,
            [[2, 3, 4, 5]],
            {},
            id="list of integers is not being validated",
        ),
        pytest.param(
            dummy_method_with_filter_input,
            [
                {
                    "condition": "OR",
                    "rules": [
                        {
                            "id": "test_age_value",
                            "field": "test_age_value",
                            "type": "int",
                            "input": "number",
                            "operator": "equal",
                            "value": 17,
                        },
                        {
                            "id": "test_pupil_reactivity_right_eye_result",
                            "field": "test_pupil_reactivity_right_eye_result",
                            "type": "string",
                            "input": "text",
                            "operator": "not_equal",
                            "value": "Nonreactive",
                        },
                    ],
                    "valid": True,
                }
            ],
            {},
            id="filters are being validated",
        ),
    ]


@pytest.mark.parametrize("func, args, kwargs", get_parametrization_list_success_cases())
def test_sql_injection_guard(func, args, kwargs):
    try:
        func(*args, **kwargs)
    except Exception as exc:
        pytest.fail(f"No exception should be raised. Exception: {exc}.")


def get_parametrization_list_exception_cases():
    return [
        pytest.param(
            dummy_method_with_one_input,
            ["notalnum!"],
            {},
            (
                ValueError,
                ".* is not alphanumeric.",
            ),
            id="invalid input, not alphanumeric",
        ),
        pytest.param(
            dummy_method_with_two_inputs,
            ["alnum1", "alnum1"],
            {},
            (
                ValueError,
                ".* is not alphabetic.",
            ),
            id="multiple inputs, one is not but should be alphabetic",
        ),
        pytest.param(
            dummy_method_with_two_inputs,
            ["alnum1"],
            {"input2": "alnum1"},
            (
                ValueError,
                ".* is not alphabetic.",
            ),
            id="multiple inputs, one is not but should be alphabetic (using kwargs)",
        ),
        pytest.param(
            dummy_method_with_one_optional_input,
            ["notalnum!"],
            {},
            (
                ValueError,
                ".* is not alphanumeric.",
            ),
            id="invalid optional input",
        ),
        pytest.param(
            dummy_method_with_one_input,
            [],
            {},
            (
                ValueError,
                "Parameter '.*' was not provided, .*",
            ),
            id="input not provided",
        ),
        pytest.param(
            dummy_method_with_one_input,
            [["not alnum"]],
            {},
            (
                ValueError,
                ".* is not alphanumeric.",
            ),
            id="invalid input passed inside list",
        ),
        pytest.param(
            dummy_method_with_one_input,
            [["alnum1", "not alnum"]],
            {},
            (
                ValueError,
                ".* is not alphanumeric.",
            ),
            id="invalid input passed inside list as 2nd element",
        ),
        pytest.param(
            dummy_method_with_one_input,
            [{"not alnum": "alnum1"}],
            {},
            (
                ValueError,
                ".* is not alphanumeric.",
            ),
            id="invalid input passed as dict key ",
        ),
        pytest.param(
            dummy_method_with_one_input,
            [{"alnum1": "not alnum"}],
            {},
            (
                ValueError,
                ".* is not alphanumeric.",
            ),
            id="invalid input passed as dict item",
        ),
        pytest.param(
            dummy_method_with_filter_input,
            [
                {
                    "condition": "OR",
                    "rules": [
                        {
                            "id": "test_age_value '); drop tables; -- ",
                            "field": "test_age_value",
                            "type": "int",
                            "input": "number",
                            "operator": "equal",
                            "value": 17,
                        },
                        {
                            "id": "test_pupil_reactivity_right_eye_result",
                            "field": "test_pupil_reactivity_right_eye_result",
                            "type": "string",
                            "input": "text",
                            "operator": "not_equal",
                            "value": "Nonreactive",
                        },
                    ],
                    "valid": True,
                }
            ],
            {},
            (
                ValueError,
                ".* is not an identifier.",
            ),
            id="invalid id in filters",
        ),
    ]


@pytest.mark.parametrize(
    "func, args, kwargs, exception", get_parametrization_list_exception_cases()
)
def test_sql_injection_guard_exceptions(func, args, kwargs, exception):
    exception_type, exception_message = exception
    with pytest.raises(exception_type, match=exception_message):
        func(*args, **kwargs)


def test_sql_injection_guard_error_when_param_name_is_not_an_argument():
    with pytest.raises(ValueError, match="Function '.*' has no argument named '.*'."):

        @sql_injection_guard("non_existing_input", isalnum)
        def dummy_method(input):
            pass


def test_is_identifier():
    identifier = "Identifier1234"
    not_identifier = "Id  123"
    try:
        isidentifier(identifier)
    except Exception as exc:
        pytest.fail(
            f"No exception should be raised. The '{identifier}' is an identifier."
        )

    with pytest.raises(ValueError, match=".* is not an identifier."):
        isidentifier(not_identifier)


def test_is_socket_address():
    socket_address = "127.0.0.1:50000"
    not_socket_address = "127.0.0.1 50000"
    try:
        is_socket_address(socket_address)
    except Exception as exc:
        pytest.fail(
            f"No exception should be raised. The '{socket_address}' is a socket address. Exception: {exc}"
        )

    with pytest.raises(ValueError, match=".* is not a socket address."):
        is_socket_address(not_socket_address)


def test_is_data_model_label():
    data_model_label = "dementia:0.1"
    not_data_model_label = "dementia:0.1  -- drop"
    try:
        isdatamodel(data_model_label)
    except Exception as exc:
        pytest.fail(
            f"No exception should be raised. The '{data_model_label}' is a data model label. Exception: {exc}"
        )

    with pytest.raises(ValueError, match=".* is not a data model label."):
        isdatamodel(not_data_model_label)


def test_is_primary_data_table():
    primary_data_table = '"dementia:0.1"."primary_data"'
    not_primary_data_table = '"dementia:0.1"."primary_data" drop'
    try:
        is_primary_data_table(primary_data_table)
    except Exception as exc:
        pytest.fail(
            f"No exception should be raised. The '{primary_data_table}' is a primary data table. Exception: {exc}"
        )

    with pytest.raises(ValueError, match=".* is not a primary data table."):
        is_primary_data_table(not_primary_data_table)


def test_get_table_schema_sql_injection_guard():
    with pytest.raises(ValueError, match=".* is not an identifier."):
        get_table_schema("not identifier")


def test_get_table_type_sql_injection_guard():
    with pytest.raises(ValueError, match=".* is not an identifier."):
        get_table_type("not identifier")


def test_get_table_names_sql_injection_guard():
    with pytest.raises(ValueError, match=".* is not alphanumeric."):
        get_table_names(None, "notalnum!")


def test_get_table_data_sql_injection_guard():
    with pytest.raises(ValueError, match=".* is not an identifier."):
        get_table_data("not identifier")


def test_get_dataset_code_per_dataset_label_sql_injection_guard():
    with pytest.raises(ValueError, match=".* is not a data model."):
        get_dataset_code_per_dataset_label("not_datamodel")


def test_get_data_model_cdes_sql_injection_guard():
    with pytest.raises(ValueError, match=".* is not a data model."):
        get_data_model_cdes("not_datamodel")


def test_drop_table_by_type_and_context_id_sql_injection_guard():
    with pytest.raises(ValueError, match=".* is not alphanumeric."):
        _drop_table_by_type_and_context_id(None, "notalnum!")


def test_drop_udfs_by_context_id_sql_injection_guard():
    with pytest.raises(ValueError, match=".* is not alphanumeric."):
        _drop_udfs_by_context_id("notalnum!")


def test_create_view_sql_injection_guard():
    with pytest.raises(ValueError, match=".* is not an identifier."):
        create_view(
            "not identifier", '"dementia:0.1"."primary_data"', ["identifier"], None
        )
    with pytest.raises(ValueError, match=".* is not a primary data table."):
        create_view(
            "identifier", '"dementia:0.1"."notprimarytable;', ["identifier"], None
        )
    with pytest.raises(ValueError, match=".* is not an identifier."):
        create_view(
            "identifier", '"dementia:0.1"."primary_data"', ["not identifier"], None
        )


def test_create_remote_table_sql_injection_guard():
    with pytest.raises(ValueError, match=".* is not an identifier."):
        create_remote_table("not identifier", None, "127.0.0.1:50000")
    with pytest.raises(ValueError, match=".* is not a socket address."):
        create_remote_table("identifier", None, "127.0.0.1 50000")


def test_create_merge_table_sql_injection_guard():
    with pytest.raises(ValueError, match=".* is not an identifier."):
        create_merge_table("not identifier", None)


def test_add_to_merge_table_sql_injection_guard():
    with pytest.raises(ValueError, match=".* is not an identifier."):
        add_to_merge_table("not identifier", [])
    with pytest.raises(ValueError, match=".* is not an identifier."):
        add_to_merge_table("identifier", ["not identifier"])


def test_validate_tables_can_be_merged_sql_injection_guard():
    with pytest.raises(ValueError, match=".* is not an identifier."):
        validate_tables_can_be_merged(["not identifier"])


def test_create_table_sql_injection_guard():
    with pytest.raises(ValueError, match=".* is not an identifier."):
        create_table("not identifier", None)
