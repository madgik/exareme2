from itertools import repeat
from string import Template


def SQL_sum_tensors(merge_table, ndims):
    udf_def = Template(
        """\
CREATE OR REPLACE
FUNCTION
${udf_name}(${call_signature})
RETURNS
TABLE(${return_signature})
LANGUAGE PYTHON
{
    import udfio
    import operator
    merge_table = udfio.make_tensor_merge_table(_columns)
    reduced = udfio.reduce_tensor_merge_table(operator.add, merge_table)
    return reduced
};"""
    )
    nodeid_column = f"CAST('$node_id' AS VARCHAR(50)) AS node_id"
    query = Template(
        f"SELECT {nodeid_column}, * FROM $udf_name((SELECT $call_args FROM $merge_table))"
    )
    dimensions = [f"dim{_}" for _ in range(ndims)]
    dimensions_wtypes = [f"{d} {t}" for d, t in zip(dimensions, repeat("INT"))]
    call_signature = ", ".join(["node_id TEXT"] + dimensions_wtypes + ["val FLOAT"])
    return_signature = ", ".join(dimensions_wtypes + ["val FLOAT"])
    call_args = ", ".join(["node_id"] + dimensions + ["val"])
    udf_def = udf_def.safe_substitute(
        call_signature=call_signature, return_signature=return_signature
    )
    query = query.safe_substitute(call_args=call_args, merge_table=merge_table)
    return udf_def, query


SQL_REDUCE_QUERIES = {
    "reduce.sum_tensors": SQL_sum_tensors,
}
