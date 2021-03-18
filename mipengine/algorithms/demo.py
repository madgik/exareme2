from typing import TypeVar

from mipengine.algorithms.udfutils import LiteralParameterT
from mipengine.algorithms.udfutils import RelationT
from mipengine.algorithms.udfutils import ScalarT
from mipengine.algorithms.udfutils import TensorT
from mipengine.algorithms.udfutils import udf

S1 = TypeVar("S1")
S2 = TypeVar("S2")


@udf
def func(x: RelationT[S1], y: RelationT[S2]) -> ScalarT(float):
    result = 5.0
    return result


T = TypeVar("T")
ND1 = TypeVar("ND1")
ND2 = TypeVar("ND2")


@udf
def tensor2(t1: TensorT[T, ND1], t2: TensorT[T, ND2]) -> TensorT[T, ND2]:
    result = t1 @ t2
    return result


@udf
def tensor1(t: TensorT[T, ND1]) -> ScalarT:
    result = 1
    return result


@udf
def table_and_literal_arguments(x: RelationT[S1], n: LiteralParameterT[int]) -> RelationT[S1]:
    result = x + n
    return result
