## UDF Generator

### Usage
Assuming there exists a file `demo.py` in `mipengine.algorithms` with the
following code
```python
# demo.py

from mipengine.algorithms.iotypes import udf
from mipengine.algorithms.iotypes import TableT

@udf
def func(x: TableT, y: TableT) -> TableT:
    result = x + y
    return result
```
then
```
>>> from mipengine.node.udfgen import generate_udf
>>> func_name = 'demo.func'
>>> udf_name = 'demo_func_1234'
>>> positional_args = [{"type": "input_table", "schema": [{"type": "int"}, {"type": "int"}], "nrows": 10}]
>>> keyword_args = {"y": {"type": "input_table", "schema": [{"type": "int"}, {"type": "int"}], "nrows": 10}}
>>> udf = generate_udf(func_name, udf_name, positional_args, keyword_args)
>>> print(udf)
CREATE OR REPLACE
FUNCTION
demo_func_1234(x0 BIGINT, x1 BIGINT, y0 BIGINT, y1 BIGINT)
RETURNS
Table(result0 BIGINT, result1 BIGINT)
LANGUAGE PYTHON
{
    from mipengine.udfgen import ArrayBundle
    x = ArrayBundle(_columns[0:2])
    y = ArrayBundle(_columns[2:4])

    # body
    result = x + y

    return as_relational_table(result)
};
```
