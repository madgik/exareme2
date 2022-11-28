# type: ignore
from typing import TypeVar

import pytest

from mipengine.datatypes import DType
from mipengine.udfgen.iotypes import DEFERRED
from mipengine.udfgen.iotypes import IOType
from mipengine.udfgen.iotypes import RelationArg
from mipengine.udfgen.iotypes import RelationType
from mipengine.udfgen.iotypes import TensorArg
from mipengine.udfgen.iotypes import literal
from mipengine.udfgen.iotypes import relation
from mipengine.udfgen.iotypes import state
from mipengine.udfgen.iotypes import tensor
from mipengine.udfgen.iotypes import transfer


def test_tensor_generic():
    DT = TypeVar("DT")
    t = tensor(dtype=DT, ndims=2)
    assert t.is_generic
    assert t.known_typeparams
    assert t.unknown_typeparams


def test_tensor_specific():
    t = tensor(dtype=int, ndims=3)
    assert not t.is_generic
    assert t.known_typeparams
    assert not t.unknown_typeparams


def test_relation_generic():
    S = TypeVar("S")
    r = relation(schema=S)
    assert r.is_generic
    assert not r.known_typeparams
    assert r.unknown_typeparams


def test_relation_specific():
    r = relation(schema=[("c1", int), ("c2", float)])
    assert isinstance(r, RelationType)


def test_relation_valid_schema_types():
    assert relation(schema=TypeVar("S"))
    assert relation(schema=DEFERRED)
    assert relation(schema=[("c1", int), ("c2", float)])
    with pytest.raises(TypeError):
        relation(schema=1)


def test_relation_valid_dtype_types():
    assert relation(schema=[("c1", int)])
    assert relation(schema=[("c1", "INT")])
    assert relation(schema=[("c1", DType.INT)])
    with pytest.raises(TypeError):
        relation(schema=[("c1", 1)])


def test_relation_invalid_schema_structure():
    with pytest.raises(TypeError):
        relation(schema=[1])


def test_literal():
    ltr = literal()
    assert isinstance(ltr, IOType)


def test_tensor_schema():
    t = tensor(dtype=DType.FLOAT, ndims=2)
    assert t.schema == [("dim0", DType.INT), ("dim1", DType.INT), ("val", DType.FLOAT)]


def test_tensor_column_names():
    t = tensor(dtype=DType.FLOAT, ndims=2)
    colnames = t.column_names(prefix="pre")
    assert colnames == ["pre_dim0", "pre_dim1", "pre_val"]


def test_relation_column_names():
    r = relation(schema=[("ci", DType.INT), ("cf", DType.FLOAT), ("cs", DType.STR)])
    colnames = r.column_names(prefix="pre")
    assert colnames == ["pre_ci", "pre_cf", "pre_cs"]


def test_transfer_schema():
    to = transfer()
    assert to.schema == [("transfer", DType.JSON)]


def test_state_schema():
    to = state()
    assert to.schema == [("state", DType.BINARY)]


class TestTensorArgsEquality:
    def test_tensor_args_names_not_equal(self):
        t1 = TensorArg("a", int, 1)
        t2 = TensorArg("b", int, 1)
        assert t1 != t2

    def test_tensor_args_dtypes_not_equal(self):
        t1 = TensorArg("a", int, 1)
        t2 = TensorArg("a", float, 1)
        assert t1 != t2

    def test_tensor_args_ndims_not_equal(self):
        t1 = TensorArg("a", int, 1)
        t2 = TensorArg("a", int, 2)
        assert t1 != t2

    def test_tensor_args_equal(self):
        t1 = TensorArg("a", int, 1)
        t2 = TensorArg("a", int, 1)
        assert t1 == t2

    def test_tensor_args_equal_different_dtype_representation(self):
        t1 = TensorArg("a", int, 1)
        t2 = TensorArg("a", DType.INT, 1)
        assert t1 == t2


class TestRelationArgsEquality:
    def test_relation_args_names_not_equal(self):
        t1 = RelationArg("a", [])
        t2 = RelationArg("b", [])
        assert t1 != t2

    def test_relation_args_schemata_names_not_equal(self):
        t1 = RelationArg("a", [("c1", int), ("c2", int)])
        t2 = RelationArg("a", [("c10", int), ("c2", int)])
        assert t1 != t2

    def test_relation_args_schemata_dtypes_not_equal(self):
        t1 = RelationArg("a", [("c1", int), ("c2", int)])
        t2 = RelationArg("a", [("c1", int), ("c2", float)])
        assert t1 != t2

    def test_relation_args_schemata_lenghts_not_equal(self):
        t1 = RelationArg("a", [("c1", int), ("c2", int)])
        t2 = RelationArg("a", [("c1", int)])
        assert t1 != t2

    def test_relation_args_equal(self):
        t1 = RelationArg("a", [("c1", int), ("c2", int)])
        t2 = RelationArg("a", [("c1", int), ("c2", int)])
        assert t1 == t2
