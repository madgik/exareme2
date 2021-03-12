## UDF Generator

### Usage
Using a UDFs defined in `logistic_regression.py` in `mipengine.algorithms`
```python
# excerpt from logistic_regression.py

DT = TypeVar("DT")  # type parameter representing datatype
ND = TypeVar("ND")  # type parameter representing number of dimensions

@udf
def tensor_expit(t: TensorT[DT, ND]) -> TensorT[DT, ND]:
    from scipy import special
    result = special.expit(t)
    return result
```
then
```
>>> from mipengine.node.udfgen import generate_udf_application_queries
>>> import mipengine.algorithms.logistic_regression
>>> from mipengine.algorithms.udfutils import TableInfo, ColumnInfo
>>> func_name = 'logistic_regression.tensor_expit'
>>> t = TableInfo(name='tens1', schema=[ColumnInfo('dim0', 'int'), ColumnInfo('val', 'float')
>>> positional_args = [t]
>>> keyword_args = {}
>>> udf, query = generate_udf_application_queries(func_name, positional_args, keyword_args)
>>> print(udf.substitute(udf_name="tensor_expit_12345"))
CREATE OR REPLACE
FUNCTION
tensor_expit_12345(tens1_dim0 INT, tens1_val float)
RETURNS
TABLE(dim0 INT, val float)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    t = udfio.from_tensor_table({n:_columns[n] for n in ['tens1_dim0', 'tens1_val']})
    from scipy import special
    result = special.expit(t)
    return udfio.as_tensor_table(numpy.array(result))
};
>>> print(query.substitute(udf_name="tensor_expit_12345", table_name="expit_result", node_id='54321'))
DROP TABLE IF EXISTS expit_result;
CREATE TABLE expit_result AS (
    SELECT 54321 AS node_id,  *
    FROM
        tensor_expit_12345(
            (
                SELECT
                    tens1.dim0, tens1.val
                FROM
                    tens1
            )
        )
);
```
