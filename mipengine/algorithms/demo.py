from mipengine.algorithms.iotypes import udf
from mipengine.algorithms.iotypes import TableT
from mipengine.algorithms.iotypes import TensorT


@udf
def func(x: TableT, y: TableT) -> TableT:
    result = x + y
    return result


@udf
def tensor3(t1: TensorT, t2: TensorT, t3: TensorT) -> TensorT:
    result = t1 @ t2 @ t3
    return result
