from string import Template


node_id_type = 'varchar(50)'
nodeid_column = f"CAST('$node_id' AS {node_id_type}) AS node_id"


def SQL_zeros1(length):
    udf_def = """\
CREATE OR REPLACE
FUNCTION
$udf_name(n INT)
RETURNS
TABLE(dim0 int, val real)
LANGUAGE PYTHON
{
    import udfio
    val = numpy.zeros((n,))
    return udfio.as_tensor_table(val)
}"""
    udf_sel = Template(f"SELECT {nodeid_column}, * FROM $udf_name($length)")
    return udf_def, udf_sel.safe_substitute(length=length)


def SQL_matrix_dot_vector(table1, table2):
    tmpl = Template(
        f"""\
SELECT
    {nodeid_column},
    t1.dim0 AS dim0,
    SUM(t1.val * t2.val) AS val
FROM $table1 AS t1, $table2 AS t2
WHERE
    t1.dim1 = t2.dim0
GROUP BY
    t1.dim0"""
    )
    return "", tmpl.safe_substitute(table1=table1, table2=table2)


def SQL_tensor1_mult(table1, table2):
    tmpl = Template(
        f"""\
SELECT
    {nodeid_column},
    t1.dim0 AS dim0,
    t1.val * t2.val AS val
FROM $table1 AS t1, $table2 AS t2
WHERE
    t1.dim0=t2.dim0"""
    )
    return "", tmpl.safe_substitute(table1=table1, table2=table2)


def SQL_tensor1_add(table1, table2):
    tmpl = Template(
        f"""\
SELECT
    {nodeid_column},
    t1.dim0 AS dim0,
    t1.val + t2.val AS val
FROM $table1 AS t1, $table2 AS t2
WHERE
    t1.dim0=t2.dim0"""
    )
    return "", tmpl.safe_substitute(table1=table1, table2=table2)


def SQL_tensor1_sub(table1, table2):
    tmpl = Template(
        f"""\
SELECT
    {nodeid_column},
    t1.dim0 AS dim0,
    t1.val - t2.val AS val
FROM $table1 AS t1, $table2 AS t2
WHERE
    t1.dim0=t2.dim0"""
    )
    return "", tmpl.safe_substitute(table1=table1, table2=table2)


def SQL_tensor1_div(table1, table2):
    tmpl = Template(
        f"""\
SELECT
    {nodeid_column},
    t1.dim0 AS dim0,
    t1.val / t2.val AS val
FROM $table1 AS t1, $table2 AS t2
WHERE
    t1.dim0=t2.dim0"""
    )
    return "", tmpl.safe_substitute(table1=table1, table2=table2)


def SQL_const_tensor1_sub(const, table):
    tmpl = Template(f"SELECT {nodeid_column}, dim0, $const - val as val from $table")
    return "", tmpl.safe_substitute(const=const, table=table)


def SQL_mat_transp_dot_diag_dot_mat(matrix, diag):
    tmpl = Template(
        f"""\
SELECT
    {nodeid_column},
    m1.dim1 AS dim0,
    m2.dim1 AS dim1,
    SUM(m1.val * d.val * m2.val) AS val
FROM $matrix AS m1, $diag AS d, $matrix as m2
WHERE
    m1.dim0 = d.dim0 AND
    m1.dim0 = m2.dim0
GROUP BY
    m1.dim1, m2.dim1"""
    )
    return "", tmpl.safe_substitute(matrix=matrix, diag=diag)


def SQL_mat_transp_dot_diag_dot_vec(matrix, diag, vec):
    tmpl = Template(
        f"""\
SELECT
    {nodeid_column},
    m.dim1 AS dim0,
    SUM(m.val * d.val * v.val) AS val
FROM $matrix AS m, $diag AS d, $vec AS v
WHERE
    m.dim0 = d.dim0 AND
    m.dim0 = v.dim0
GROUP BY
    m.dim1"""
    )
    return "", tmpl.safe_substitute(matrix=matrix, diag=diag, vec=vec)


SQL_LINALG_QUERIES = {
    "sql.zeros1": SQL_zeros1,
    "sql.matrix_dot_vector": SQL_matrix_dot_vector,
    "sql.tensor1_mult": SQL_tensor1_mult,
    "sql.tensor1_add": SQL_tensor1_add,
    "sql.tensor1_sub": SQL_tensor1_sub,
    "sql.tensor1_div": SQL_tensor1_div,
    "sql.const_tensor1_sub": SQL_const_tensor1_sub,
    "sql.mat_transp_dot_diag_dot_mat": SQL_mat_transp_dot_diag_dot_mat,
    "sql.mat_transp_dot_diag_dot_vec": SQL_mat_transp_dot_diag_dot_vec,
}
