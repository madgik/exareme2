from typing import TypeVar

from mipengine.algorithms.udfiotypes import udf
from mipengine.algorithms.udfiotypes import RelationT
from mipengine.algorithms.udfiotypes import RelationT
from mipengine.algorithms.udfiotypes import TensorT
from mipengine.algorithms.udfiotypes import ScalarT


S1 = TypeVar("S1")
S2 = TypeVar("S2")


@udf
def func(x: RelationT[S1], y: RelationT[S2]) -> RelationT[S1]:
    result = x
    return result


T = TypeVar("T")
ND1 = TypeVar("ND1")
ND2 = TypeVar("ND2")


@udf
def tensor3(t1: TensorT[T, ND1], t2: TensorT[T, ND2]) -> TensorT[T, ND2]:
    result = t1 @ t2
    return result


@udf
def tensor1(t: TensorT[T, ND1]) -> ScalarT:
    result = 1
    return result
