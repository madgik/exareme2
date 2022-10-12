from collections import namedtuple

import pytest

from mipengine.node.monetdb_interface.guard import InvalidSQLParameter
from mipengine.node.monetdb_interface.guard import is_datamodel
from mipengine.node.monetdb_interface.guard import is_list_of_identifiers
from mipengine.node.monetdb_interface.guard import is_primary_data_table
from mipengine.node.monetdb_interface.guard import is_socket_address
from mipengine.node.monetdb_interface.guard import is_valid_filter
from mipengine.node.monetdb_interface.guard import is_valid_literal_value
from mipengine.node.monetdb_interface.guard import is_valid_request_id
from mipengine.node.monetdb_interface.guard import is_valid_table_schema
from mipengine.node.monetdb_interface.guard import sql_injection_guard


@pytest.mark.parametrize(
    "string",
    [
        "0.0.0.0:1",
        "192.168.1.1:50000",
        "255.255.255.255:65535",
    ],
)
def test_is_socket_address_valid(string):
    assert is_socket_address(string)


@pytest.mark.parametrize(
    "string",
    [
        "0.0.0.0:0",
        "192.168.1:50000",
        " 192.168.1.1:50000",
        "192.168.1.1:50000 ",
        "192.168.1.1",
        "255.255.255.256:65535",
        "255.255.255.255:65536",
    ],
)
def test_is_socket_address_invalid(string):
    assert not is_socket_address(string)


@pytest.mark.parametrize(
    "string",
    [
        "datamodel:0",
        "datamodel:0.1",
        "datamodel:alpha",
        "datamodel:alpha_beta",
        "data_model_1:0.1",
    ],
)
def test_is_datamodel_valid(string):
    assert is_datamodel(string)


@pytest.mark.parametrize(
    "string",
    [
        "datamodel",
        "datamodel:",
        ":0.1",
        "datamodel:.0",
        "datamodel:0.",
    ],
)
def test_is_datamodel_invalid(string):
    assert not is_datamodel(string)


@pytest.mark.parametrize(
    "string",
    [
        "data_table",
        '"datamodel:0.1".data_table',
        '"datamodel:0.1"."Data_Table"',
    ],
)
def test_is_primary_data_table_valid(string):
    assert is_primary_data_table(string)


@pytest.mark.parametrize(
    "string",
    [
        "data table",
        '"datamodel:0.1".Data_Table',
        '"datamodel:0.1".',
        '"datamodel:0.1"."Data Table"',
    ],
)
def test_is_primary_data_table_invalid(string):
    assert not is_primary_data_table(string)


def test_is_list_of_identifiers():
    assert is_list_of_identifiers(["name_1", "name_2"])
    assert not is_list_of_identifiers(["name.1", "name_2"])


def test_is_valid_filter():
    assert is_valid_filter({"rules": [{"id": "name1"}, {"rules": [{"id": "name2"}]}]})
    assert not is_valid_filter(
        {"rules": [{"id": "name1"}, {"rules": [{"id": "name.2"}]}]}
    )


def test_is_valid_table_schema():
    Column = namedtuple("Column", "name")
    Schema = namedtuple("Schema", "columns")

    valid_schema = Schema(columns=[Column(name="name_1"), Column(name="name_2")])
    invalid_schema = Schema(columns=[Column(name="name_1"), Column(name="name.2")])

    assert is_valid_table_schema(valid_schema)
    assert not is_valid_table_schema(invalid_schema)


@pytest.mark.parametrize(
    "string",
    [
        "not identifier",
        "2343-2342342-",
        "--",
        "identifieranduuid-89aace55-60e8-4b29-958b-84cca8785120",
    ],
)
def test_is_valid_request_id_invalid(string):
    assert not is_valid_request_id(string)


def test_is_valid_request_id_valid():
    assert is_valid_request_id("identifier2432342")
    assert is_valid_request_id("89aace55-60e8-4b29-958b-84cca8785120")
    assert is_valid_request_id("89AACE55-60e8-4b29-958b-84cca8785120")


def test_sql_injection_guard__validate_posarg():
    @sql_injection_guard(a=str.isalpha)
    def f(a):
        pass

    try:
        f("a")
    except InvalidSQLParameter as exc:
        pytest.fail(f"Unexpected exception {exc}")


def test_sql_injection_guard__validate_kwarg():
    @sql_injection_guard(a=str.isalpha)
    def f(a):
        pass

    try:
        f(a="a")
    except InvalidSQLParameter as exc:
        pytest.fail(f"Unexpected exception {exc}")


def test_sql_injection_guard__validate_default():
    @sql_injection_guard(a=str.isalpha)
    def f(a="a"):
        pass

    try:
        f()
    except InvalidSQLParameter as exc:
        pytest.fail(f"Unexpected exception {exc}")


def test_sql_injection_guard__missing_validator():
    def f(a, b):
        pass

    with pytest.raises(ValueError):
        sql_injection_guard(a=None)(f)


def test_sql_injection_guard__missing_param():
    def f(a):
        pass

    with pytest.raises(ValueError):
        sql_injection_guard(a=None, b=None)(f)


def test_sql_injection_guard__raise_invalid_parameter():
    @sql_injection_guard(a=str.isalnum)
    def f(a):
        pass

    with pytest.raises(InvalidSQLParameter):
        f(a="!")


@pytest.mark.parametrize(
    "val",
    [
        1,
        "a",
        ["a", "b"],
        {"a": "b", "c": 1},
        [{"a": "b"}],
        {"a": ["b"]},
    ],
)
def test_is_valid_literal_value__valid(val):
    assert is_valid_literal_value(val)


@pytest.mark.parametrize(
    "val",
    [
        "};",
        ["}", ";"],
        {"}": ";"},
        [{"}": ";"}],
        {"}": [";"]},
    ],
)
def test_is_valid_literal_value__invalid(val):
    assert not is_valid_literal_value(val)
