-- zeros1
CREATE OR REPLACE
FUNCTION
zeros1(len INT)
RETURNS
TABLE(dim0 int, val float)
LANGUAGE PYTHON
{
    import udfio
    val = numpy.zeros((len,))
    return udfio.as_tensor_table(val)
};

-- matrix_dot_vector
SELECT
    t1.dim0 AS dim0,
    SUM(t1.val * t2.val) AS val
FROM t1, t2
WHERE
    t1.dim1 = t2.dim0
GROUP BY
    t1.dim0;

-- tensor1_mult
SELECT
    t1.dim0 AS dim0,
    t1.val * t1.val AS val
FROM t1, t2
WHERE
    t1.dim0=t2.dim0;

-- tensor1_add
SELECT
    t1.dim0 AS dim0,
    t1.val + t1.val AS val
FROM t1, t2
WHERE
    t1.dim0=t2.dim0;

-- tensor1_sub
SELECT
    t1.dim0 AS dim0,
    t1.val - t1.val AS val
FROM t1, t2
WHERE
    t1.dim0=t2.dim0;

-- tensor1_div
SELECT
    t1.dim0 AS dim0,
    t1.val / t1.val AS val
FROM t1, t2
WHERE
    t1.dim0=t2.dim0;

-- const_tensor_sub1
SELECT dim0, 5-val from t1;

-- mat_transp_dot_diag_dot_mat Mji Dj Mjk
SELECT
    m1.dim1 AS dim0,
    m2.dim1 AS dim1,
    SUM(m1.val * d.val * m2.val) AS val
FROM m as m1, d, m as m2
WHERE
    m1.dim0 = d.dim0 AND
    m1.dim0 = m2.dim0
GROUP BY
    m1.dim1, m2.dim1;

-- mat_transp_dot_diag_dot_vec Mji Dj Vj
SELECT
    m.dim1 AS dim0,
    SUM(m.val * d.val * v.val) AS val
FROM m, d, v
WHERE
    m.dim0 = d.dim0 AND
    m.dim0 = v.dim0
GROUP BY
    m.dim1:
