## UDF Generator

### Usage
Assuming there exists a file `demo.py` in `mipengine.algorithms` with the
following code
```python
# demo.py

from mipengine.algorithms.iotypes import udf
from mipengine.algorithms.iotypes import TableT
from mipengine.algorithms.iotypes import TensorT


@udf
def func(x: TableT) -> TensorT:
    result = x.T @ x
    return result
```
then
```
>>> from mipengine.node.udfgen import generate_udf
>>> func_name = 'demo.func'
>>> udf_name = 'demo_func_1234'
>>> input_tables = [{'schema': [{'type': int}, {'type': int}], 'nrows': 100}]
>>> loopback_tables = []
>>> literal_params = {}
>>> udf = generate_udf(func_name, udf_name, input_tables, loopback_tables, literal_params)
>>> print(udf)
CREATE OR REPLACE
FUNCTION
demo_func_1234(x0 BIGINT, x1 BIGINT)
RETURNS
Table(result0 BIGINT, result1 BIGINT)
LANGUAGE PYTHON
{
    from mipengine.udfgen import ArrayBundle
    x = ArrayBundle(_columns[0:2])

    # body
    result = x.T @ x

    return as_tensor_table(result)
};
```
