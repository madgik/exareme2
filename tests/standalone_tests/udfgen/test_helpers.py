# type: ignore
import pytest

from exareme2.udfgen.helpers import mapping_inverse
from exareme2.udfgen.helpers import mappings_coincide
from exareme2.udfgen.helpers import merge_mappings_consistently
from exareme2.udfgen.helpers import recursive_repr


class TestMergeMappings:
    def test_merge_mappings_no_overlap(self):
        mappings = [{"a": 1, "b": 2}, {"c": 3, "d": 4}]
        merged = merge_mappings_consistently(mappings)
        assert merged == {"a": 1, "b": 2, "c": 3, "d": 4}

    def test_merge_mappings_map_overlap(self):
        mappings = [{"a": 1, "b": 2}, {"c": 3, "a": 1}]
        merged = merge_mappings_consistently(mappings)
        assert merged == {"a": 1, "b": 2, "c": 3}

    def test_merge_mappings_error_missmatch(self):
        mappings = [{"a": 1, "b": 2}, {"c": 3, "a": 10}]
        with pytest.raises(ValueError):
            merge_mappings_consistently(mappings)


class TestMappingInverse:
    def test_mapping_inverse_valid(self):
        mapping = {"a": 1, "b": 2}
        assert mapping_inverse(mapping) == {1: "a", 2: "b"}

    def test_mapping_inverse_nvalid(self):
        mapping = {"a": 1, "b": 1}
        with pytest.raises(ValueError):
            mapping_inverse(mapping)


class TestMappingsCoincide:
    def test_mappings_coincide_true(self):
        map1 = {"a": 1, "b": 2}
        map2 = {"a": 1, "c": 3}
        assert mappings_coincide(map1, map2) is True

    def test_mappings_coincide_false(self):
        map1 = {"a": 1, "b": 2}
        map2 = {"a": 10, "c": 3}
        assert mappings_coincide(map1, map2) is False


def test_constructor_repr():
    class A:
        def __init__(self, a, b) -> None:
            self.a, self.b = a, b

        __repr__ = recursive_repr

    a = A(1, 2)
    assert repr(a) == "A(a=1,b=2)"
