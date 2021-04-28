import pytest

from mipengine.common.validators import validate_sql_params
from mipengine.common.validators import _validate_sql_param
from mipengine.common.validators import _validate_socket_address


@pytest.mark.parametrize(
    "args, kwargs",
    [
        (["hello", "world"], {}),
        (["1234", "_name", {"table_name": "ThisTable42"}, ["nested_list"]], {}),
        ([], {"nested": {"dict": {"name": "Bob"}}}),
        ([[["nested"], ["1list"]]], {"nested": {"dict": {"name": "Bob"}}}),
    ],
)
def test_validate_sql_params_wrapper(args, kwargs):
    def returns_true(*args, **kwargs):
        return args, kwargs

    wrapper = validate_sql_params(returns_true)
    assert wrapper(*args, **kwargs)


@pytest.mark.parametrize(
    "args, kwargs",
    [
        ([12345, "world"], {}),
        ([{";nono": "yes"}], {}),
        (["1234", "_name", {"table_name": "--ThisTable42"}, ["nested_list"]], {}),
        ([], {"nested": {"dict": {"name": "evil'ex"}}}),
        ([[["evil'ex"]]], {"nested": {"dict": {"name": "Bob"}}}),
    ],
)
def test_validate_sql_params_wrapper_error(args, kwargs):
    def returns_true(*args, **kwargs):
        return args, kwargs

    wrapper = validate_sql_params(returns_true)
    with pytest.raises(ValueError):
        wrapper(*args, **kwargs)


@pytest.mark.parametrize(
    "arg",
    ["12345", "hello" "__name1234", "this_is_a_name_", "2AF46BE3", "127.0.0.1:1000"],
)
def test_validate_param(arg):
    _validate_sql_param(arg)


@pytest.mark.parametrize(
    "arg",
    [
        "Robert'); DROP TABLE students;--",
        "' or 1=1; --",
        "evil’ex",
        "--123--" "#hashtag",
        "myname@gmail.com",
        "**Important**",
    ],
)
def test_validate_param_error(arg):
    with pytest.raises(ValueError):
        _validate_sql_param(arg)


@pytest.mark.parametrize(
    "arg",
    [
        "192.168.1.51:5000",
        "127.0.0.1:888",
        "0.0.0.0:0",
        "256.256.256.256:65535",
    ],
)
def test_validate_socket_address(arg):
    assert _validate_socket_address(arg)


@pytest.mark.parametrize(
    "arg",
    [
        ".199.23.402:123",
        "123j.123.123.123:12345",
        "...:",
        "hello",
        "-1.23.53.45:-100",
        "12b.43.53h.10:yes",
        "127.00.1:1000",
    ],
)
def test_validate_socket_address_error(arg):
    assert not _validate_socket_address(arg)
