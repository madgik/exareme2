from mipengine.algorithms.iotypes import udf
from mipengine.algorithms.iotypes import TableT

@udf
def func(x: TableT, y: TableT) -> TableT:
    result = x + y
    return result
