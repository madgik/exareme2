from typing import TypeVar

from mipengine.algorithms.iotypes import udf
from mipengine.algorithms.iotypes import RelationT
from mipengine.algorithms.iotypes import RelationFromSchemaT
from mipengine.algorithms.iotypes import TensorT
from mipengine.algorithms.iotypes import ScalarT


S1 = TypeVar("S1")
S2 = TypeVar("S2")


@udf
def func(
    x: RelationFromSchemaT[S1], y: RelationFromSchemaT[S2]
) -> RelationFromSchemaT[S1]:
    result = x
    return result


T = TypeVar("T")
R1 = TypeVar("R1")
R2 = TypeVar("R2")
R3 = TypeVar("R3")
C1 = TypeVar("C1")
C2 = TypeVar("C2")
C3 = TypeVar("C3")


@udf
def tensor3(t1: TensorT[T, R1, C1], t2: TensorT[T, R2, C2]) -> TensorT[T, R1, C2]:
    result = t1 @ t2
    return result


@udf
def tensor1(t: TensorT) -> ScalarT:
    result = 1
    return result
