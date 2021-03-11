DROP TABLE IF EXISTS tens1;
CREATE TABLE tens1(dim0 INT, val FLOAT);
INSERT INTO tens1 VALUES (0, 0.50);
INSERT INTO tens1 VALUES (1, 0.73);
INSERT INTO tens1 VALUES (2, 0.93);
INSERT INTO tens1 VALUES (3, 0.111);

DROP TABLE IF EXISTS tens2;
CREATE TABLE tens2(dim0 INT, val FLOAT);
INSERT INTO tens2 VALUES (0, 0.55);
INSERT INTO tens2 VALUES (1, 0.93);
INSERT INTO tens2 VALUES (2, 0.113);
INSERT INTO tens2 VALUES (3, 0.81);

DROP TABLE IF EXISTS tens3;
CREATE TABLE tens3(dim0 INT, dim1 INT, val FLOAT);
INSERT INTO tens3 VALUES (0, 0, 5.5);
INSERT INTO tens3 VALUES (0, 1, 9.3);
INSERT INTO tens3 VALUES (1, 0, 11.3);
INSERT INTO tens3 VALUES (1, 1, 8.1);

DROP TABLE IF EXISTS vector;
CREATE TABLE vector(dim0 INT, val FLOAT);
INSERT INTO vector VALUES (0, 2.57);
INSERT INTO vector VALUES (1, 5.93);

DROP TABLE IF EXISTS diag;
CREATE TABLE diag(dim0 INT, val FLOAT);
INSERT INTO diag VALUES (0, 1.55);
INSERT INTO diag VALUES (1, 3.93);

DROP TABLE IF EXISTS matrix;
CREATE TABLE matrix(dim0 INT, dim1 INT, val FLOAT);
INSERT INTO matrix VALUES (0, 0, 5.5);
INSERT INTO matrix VALUES (0, 1, 9.3);
INSERT INTO matrix VALUES (1, 0, 11.3);
INSERT INTO matrix VALUES (1, 1, 8.1);

CREATE OR REPLACE
FUNCTION
tensor_expit(tens1_dim0 INT, tens1_val float)
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
DROP TABLE IF EXISTS expit_result;
CREATE TABLE expit_result AS (
    SELECT *
    FROM
        tensor_expit(
            (
                SELECT
                    tens1.dim0, tens1.val
                FROM
                    tens1
            )
        )
);
SELECT * FROM expit_result;

CREATE OR REPLACE
FUNCTION
logistic_loss(tens1_dim0 INT, tens1_val float, tens2_dim0 INT, tens2_val float)
RETURNS
FLOAT
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    v1 = udfio.from_tensor_table({n:_columns[n] for n in ['tens1_dim0', 'tens1_val']})
    v2 = udfio.from_tensor_table({n:_columns[n] for n in ['tens2_dim0', 'tens2_val']})
    from scipy import special
    ll = numpy.sum(special.xlogy(v1, v2) + special.xlogy(1 - v1, 1 - v2))
    return ll
};
DROP TABLE IF EXISTS logistic_loss_result;
CREATE TABLE logistic_loss_result AS (
    SELECT
        logistic_loss(tens1.dim0, tens1.val, tens2.dim0, tens2.val)
    FROM
        tens1, tens2
    WHERE
        tens1.dim0=tens2.dim0
);
SELECT * FROM logistic_loss_result;

CREATE OR REPLACE
FUNCTION
tensor_max_abs_diff(tens1_dim0 INT, tens1_val float, tens2_dim0 INT, tens2_val float)
RETURNS
FLOAT
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    t1 = udfio.from_tensor_table({n:_columns[n] for n in ['tens1_dim0', 'tens1_val']})
    t2 = udfio.from_tensor_table({n:_columns[n] for n in ['tens2_dim0', 'tens2_val']})
    result = numpy.max(numpy.abs(t1 - t2))
    return result
};
DROP TABLE IF EXISTS tensor_max_abs_diff_result;
CREATE TABLE tensor_max_abs_diff_result AS (
    SELECT
        tensor_max_abs_diff(tens1.dim0, tens1.val, tens2.dim0, tens2.val)
    FROM
        tens1, tens2
    WHERE
        tens1.dim0=tens2.dim0
);
SELECT * FROM tensor_max_abs_diff_result;

CREATE OR REPLACE
FUNCTION
mat_inverse(tens3_dim0 INT, tens3_dim1 INT, tens3_val float)
RETURNS
TABLE(dim0 INT, dim1 INT, val float)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    M = udfio.from_tensor_table({n:_columns[n] for n in ['tens3_dim0', 'tens3_dim1', 'tens3_val']})
    minv = numpy.linalg.inv(M)
    return udfio.as_tensor_table(numpy.array(minv))
};
DROP TABLE IF EXISTS mat_inverse_result;
CREATE TABLE mat_inverse_result AS (
    SELECT *
    FROM
        mat_inverse(
            (
                SELECT
                    tens3.dim0, tens3.dim1, tens3.val
                FROM
                    tens3
            )
        )
);
SELECT * FROM mat_inverse_result;

CREATE OR REPLACE
FUNCTION
zeros1(n INT)
RETURNS
TABLE(dim0 int, val float)
LANGUAGE PYTHON
{
    import udfio
    val = numpy.zeros((n,))
    return udfio.as_tensor_table(val)
};
DROP TABLE IF EXISTS zeros1_result;
CREATE TABLE zeros1_result AS (SELECT * FROM zeros1(5));
SELECT * FROM zeros1_result;

DROP TABLE IF EXISTS matrix_dot_vector_results;
CREATE TABLE matrix_dot_vector_results AS (
    SELECT
        t1.dim0 AS dim0,
        SUM(t1.val * t2.val) AS val
    FROM matrix AS t1, vector AS t2
    WHERE
        t1.dim1 = t2.dim0
    GROUP BY
        t1.dim0
);
SELECT * FROM matrix_dot_vector_results;

DROP TABLE IF EXISTS tensor_mult_result;
CREATE TABLE tensor_mult_result AS (
    SELECT
        t1.dim0 AS dim0,
        t1.val * t1.val AS val
    FROM tens1 AS t1, tens1 AS t2
    WHERE
        t1.dim0=t2.dim0
);
SELECT * FROM tensor_mult_result;

DROP TABLE IF EXISTS tensor_add_result;
CREATE TABLE tensor_add_result AS (
    SELECT
        t1.dim0 AS dim0,
        t1.val + t1.val AS val
    FROM tens1 AS t1, tens1 AS t2
    WHERE
        t1.dim0=t2.dim0
);
SELECT * FROM tensor_add_result;

DROP TABLE IF EXISTS tensor_sub_result;
CREATE TABLE tensor_sub_result AS (
    SELECT
        t1.dim0 AS dim0,
        t1.val - t1.val AS val
    FROM tens1 AS t1, tens1 AS t2
    WHERE
        t1.dim0=t2.dim0
);
SELECT * FROM tensor_sub_result;

DROP TABLE IF EXISTS tensor_div_result;
CREATE TABLE tensor_div_result AS (
    SELECT
        t1.dim0 AS dim0,
        t1.val / t1.val AS val
    FROM tens1 AS t1, tens1 AS t2
    WHERE
        t1.dim0=t2.dim0
);
SELECT * FROM tensor_div_result;

DROP TABLE IF EXISTS const_tensor_sub1_result;
CREATE TABLE const_tensor_sub1_result AS (SELECT dim0, 1-val from tens1);
SELECT * FROM const_tensor_sub1_result;

DROP TABLE IF EXISTS mat_transp_dot_diag_dot_mat_result;
CREATE TABLE mat_transp_dot_diag_dot_mat_result AS (
    SELECT
        m1.dim1 AS dim0,
        m2.dim1 AS dim1,
        SUM(m1.val * d.val * m2.val) AS val
    FROM matrix AS m1, vector AS d, matrix as m2
    WHERE
        m1.dim0 = d.dim0 AND
        m1.dim0 = m2.dim0
    GROUP BY
        m1.dim1, m2.dim1
);
SELECT * FROM mat_transp_dot_diag_dot_mat_result;

DROP TABLE IF EXISTS mat_transp_dot_diag_dot_vec_result;
CREATE TABLE mat_transp_dot_diag_dot_vec_result AS (
    SELECT
        m.dim1 AS dim0,
        SUM(m.val * d.val * v.val) AS val
    FROM matrix AS m, diag AS d, vector AS v
    WHERE
        m.dim0 = d.dim0 AND
        m.dim0 = v.dim0
    GROUP BY
        m.dim1
);
SELECT * FROM mat_transp_dot_diag_dot_vec_result;
