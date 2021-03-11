
DROP TABLE IF EXISTS tens1;
CREATE TABLE tens1(dim0 INT, val FLOAT);
INSERT INTO tens1 VALUES (0, 5.0);
INSERT INTO tens1 VALUES (1, 7.3);
INSERT INTO tens1 VALUES (2, 9.3);
INSERT INTO tens1 VALUES (3, 11.1);

DROP TABLE IF EXISTS tens2;
CREATE TABLE tens2(dim0 INT, val FLOAT);
INSERT INTO tens2 VALUES (0, 5.5);
INSERT INTO tens2 VALUES (1, 9.3);
INSERT INTO tens2 VALUES (2, 11.3);
INSERT INTO tens2 VALUES (3, 8.1);

DROP TABLE IF EXISTS tens3;
CREATE TABLE tens3(dim0 INT, dim1 INT, val FLOAT);
INSERT INTO tens3 VALUES (0, 0, 5.5);
INSERT INTO tens3 VALUES (0, 1, 9.3);
INSERT INTO tens3 VALUES (1, 0, 11.3);
INSERT INTO tens3 VALUES (1, 1, 8.1);

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

