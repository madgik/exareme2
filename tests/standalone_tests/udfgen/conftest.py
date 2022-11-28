import pytest

from mipengine.udfgen import udf


@pytest.fixture(autouse=True)
def clear_udf_registry():
    yield
    udf.registry = {}
