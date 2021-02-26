from mipengine.algorithms.iotypes import udf
from mipengine.algorithms.iotypes import TableT
from mipengine.algorithms.iotypes import TensorT


@udf
def func(x: TableT) -> TensorT:
    result = x.T @ x
    return result
