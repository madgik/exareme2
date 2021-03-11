-- X: matrix, V: vector
--
-- X:                     V:
-- +------+------+-----+  +------+------+
-- | dim0 | dim1 | val |  | dim0 | val  |
-- +------+------+-----+  +------+------+
-- | 0    | 0    | 1.0 |  | 0    | 1.0  |
-- | 0    | 1    | 2.0 |  | 1    | 2.0  |
-- | 0    | 2    | 3.0 |  | 2    | 3.0  |
-- | 1    | 0    | 4.0 |  +------+------+
-- | 1    | 1    | 5.0 |
-- | 1    | 2    | 6.0 |
-- | 2    | 0    | 7.0 |
-- | 2    | 1    | 8.0 |
-- | 2    | 2    | 9.0 |
-- +------+------+-----+

DROP TABLE IF EXISTS X;
CREATE TABLE X(dim0 int, dim1 int, val float);
INSERT INTO X VALUES (0, 0, 1);
INSERT INTO X VALUES (0, 1, 2);
INSERT INTO X VALUES (0, 2, 3);
INSERT INTO X VALUES (1, 0, 4);
INSERT INTO X VALUES (1, 1, 5);
INSERT INTO X VALUES (1, 2, 6);
INSERT INTO X VALUES (2, 0, 7);
INSERT INTO X VALUES (2, 1, 8);
INSERT INTO X VALUES (2, 2, 9);

DROP TABLE IF EXISTS V;
CREATE TABLE V(dim0 int, val float);
INSERT INTO V VALUES (0, 1);
INSERT INTO V VALUES (1, 2);
INSERT INTO V VALUES (2, 3);

-- Matrix vector dot product: X @ V or X_ij V_j
SELECT
    X.dim0 AS dim0,
    SUM(X.val * V.val) AS val
FROM X, V
WHERE
    X.dim1 = V.dim0
GROUP BY
    X.dim0;


DROP TABLE IF EXISTS Y;
CREATE TABLE Y(dim0 int, dim1 int, val float);
INSERT INTO Y VALUES (0, 0, 1);
INSERT INTO Y VALUES (0, 1, 2);
INSERT INTO Y VALUES (0, 2, 3);
INSERT INTO Y VALUES (1, 0, 4);
INSERT INTO Y VALUES (1, 1, 5);
INSERT INTO Y VALUES (1, 2, 6);
INSERT INTO Y VALUES (2, 0, 7);
INSERT INTO Y VALUES (2, 1, 8);
INSERT INTO Y VALUES (2, 2, 9);

-- Matrix matrix dot product: X @ Y or X_ij Y_jk
SELECT
    X.dim0 AS dim0,
    Y.dim1 AS dim1,
    SUM(X.val * Y.val) AS val
FROM X, Y
WHERE
    X.dim1 = Y.dim0
GROUP BY
    X.dim0, Y.dim1;

-- Matrix transposition with dot product: X.T @ Y or X_ji Y_jk
SELECT
    X.dim1 AS dim0,
    Y.dim1 AS dim1,
    SUM(X.val * Y.val) AS val
FROM X, Y
WHERE
    X.dim0 = Y.dim0
GROUP BY
    X.dim1, Y.dim1;

-- Transpose: X.T :: X_ij -> X_ji
SELECT
    t1.dim1 AS dim0,
    t1.dim0 AS dim1,
    t1.val AS val
FROM X AS t1
ORDER BY dim0, dim1;

-- Trace: X_ii
SELECT
    SUM(t1.val) AS val
FROM X AS t1
WHERE
    t1.dim0 = t1.dim1;

-- Outer product: V_i V_j -> M_ij
SELECT
    t1.dim0 as dim0,
    t2.dim0 as dim1,
    SUM(t1.val * t2.val) as val
FROM V AS t1, V AS t2
GROUP BY t1.dim0, t2.dim0;


DROP TABLE IF EXISTS T1;
CREATE TABLE T1(dim0 int, dim1 int, dim2 int, val int);
INSERT INTO T1 VALUES (0, 0, 0, 1);
INSERT INTO T1 VALUES (0, 0, 1, 2);
INSERT INTO T1 VALUES (0, 1, 0, 3);
INSERT INTO T1 VALUES (0, 1, 1, 4);
INSERT INTO T1 VALUES (1, 0, 0, 5);
INSERT INTO T1 VALUES (1, 0, 1, 6);
INSERT INTO T1 VALUES (1, 1, 0, 7);
INSERT INTO T1 VALUES (1, 1, 1, 8);

DROP TABLE IF EXISTS T2;
CREATE TABLE T2(dim0 int, dim1 int, val float);
INSERT INTO T2 VALUES (0, 0, 1);
INSERT INTO T2 VALUES (0, 1, 2);
INSERT INTO T2 VALUES (1, 0, 3);
INSERT INTO T2 VALUES (1, 1, 4);

-- Tensor contraction T1_ijk T1_kl -> T3_ijl
SELECT
    T1.dim0 AS dim0,
    T1.dim1 AS dim1,
    T2.dim1 AS dim2,
    SUM(T1.val * T2.val) AS val
FROM T1, T2
WHERE
    T1.dim2 = T2.dim0
GROUP BY
    T1.dim0, T1.dim1, T2.dim1;

-- Tensor contraction T1_ijk T1_jk -> T3_i
SELECT
    T1.dim0 AS dim0,
    SUM(T1.val * T2.val) AS val
FROM T1, T2
WHERE
    T1.dim1 = T2.dim0 and T1.dim2 = T2.dim1
GROUP BY
    T1.dim0;

-- Tensor contraction T1_ijk T1_ji -> T3_k
SELECT
    T1.dim2 AS dim0,
    SUM(T1.val * T2.val) AS val
FROM T1, T2
WHERE
    T1.dim0 = T2.dim1 and T1.dim1 = T2.dim0
GROUP BY
    T1.dim2;

-- Self contraction T1_ijj -> T3_i
SELECT
    p1.dim0 AS dim0,
    SUM(p1.val) AS val
FROM T1 as p1
WHERE
    p1.dim1 = p1.dim2
GROUP BY
    p1.dim0;

-- Tensor transposition T1_ijk -> T1_jik
SELECT
    t.dim1 AS dim0,
    t.dim0 AS dim1,
    t.dim2 AS dim2,
    t.val AS val
FROM T1 as t
ORDER BY dim0, dim1, dim2;

-- Tensor contraction and transposition T1_ijk T2_kl -> T3_ijl -> T3_jil
SELECT
    t.dim1 AS dim0,
    t.dim0 AS dim1,
    t.dim2 AS dim2,
    t.val AS val
FROM (
    SELECT
        T1.dim0 AS dim0,
        T1.dim1 AS dim1,
        T2.dim1 AS dim2,
        SUM(T1.val * T2.val) AS val
    FROM T1, T2
    WHERE
        T1.dim2 = T2.dim0
    GROUP BY
        T1.dim0, T1.dim1, T2.dim1
) AS t
ORDER BY dim0, dim1, dim2;

----------------------------------------------------------------------------
-- diag Matrix diagonal -> vector
SELECT
    t1.dim0,
    t1.val AS val
FROM X AS t1
WHERE
    t1.dim0 = t1.dim1;

-- undiag Vector -> diagonal matrix
SELECT
    t1.dim0 as dim0,
    t2.dim0 as dim1,
    CASE
    WHEN t1.dim0 = t2.dim0
        THEN t1.val
    ELSE
        0
    END AS val
FROM V AS t1, V AS t2;
----------------------------------------------------------------------------
-- Select tests

CREATE OR REPLACE
FUNCTION
yes(x0 BIGINT, x1 BIGINT, x2 BIGINT, x3 BIGINT, y0 BIGINT, y1 BIGINT, y2 BIGINT, y3 BIGINT)
RETURNS
TABLE(t0 int, t1 int, t2 int, t3 int)
LANGUAGE PYTHON
{
    import numpy as np
    x = np.array([x0, x1, x2, x3])
    y = np.array([y0, y1, y2, y3])
    result = x + y
    return x + y
};


SELECT * FROM
    yes(
        (SELECT
            t1.dim0, t1.dim1, t1.dim2, t1.val,
            t2.dim0, t2.dim1, t2.dim2, t2.val
        FROM t1, t2
        WHERE
            t1.dim0=t2.dim0 AND
            t1.dim1=t2.dim1 AND
            t1.dim2=t2.dim2
        )
    );

--------------------------------------------------
-- Larger matrix

DROP TABLE IF EXISTS x2;
CREATE TABLE x2(dim0 int, dim1 int, val int);
INSERT INTO x2 VALUES (0, 0, 0 );
INSERT INTO x2 VALUES (0, 1, 1 );
INSERT INTO x2 VALUES (0, 2, 2 );
INSERT INTO x2 VALUES (0, 3, 3 );
INSERT INTO x2 VALUES (1, 0, 4 );
INSERT INTO x2 VALUES (1, 1, 5 );
INSERT INTO x2 VALUES (1, 2, 6 );
INSERT INTO x2 VALUES (1, 3, 7 );
INSERT INTO x2 VALUES (2, 0, 8 );
INSERT INTO x2 VALUES (2, 1, 9 );
INSERT INTO x2 VALUES (2, 2, 10);
INSERT INTO x2 VALUES (2, 3, 11);
INSERT INTO x2 VALUES (3, 0, 12);
INSERT INTO x2 VALUES (3, 1, 13);
INSERT INTO x2 VALUES (3, 2, 14);
INSERT INTO x2 VALUES (3, 3, 15);

DROP TABLE IF EXISTS v2;
CREATE TABLE v2(dim0 int, val int);
INSERT INTO v2 VALUES (0, 0);
INSERT INTO v2 VALUES (1, 1);
INSERT INTO v2 VALUES (2, 2);
INSERT INTO v2 VALUES (3, 3);

-- Matrix vector dot product: X @ V
SELECT
    m.dim0 AS dim0,
    SUM(m.val * v.val) AS val
FROM x2 as m, v2 as v
WHERE
    m.dim1 = v.dim0
GROUP BY
    m.dim0
ORDER BY
    dim0;

-------------------------------------------------------
-- zeros1
CREATE OR REPLACE
FUNCTION
zeros1(*)
RETURNS
TABLE(dim0 int, val float)
LANGUAGE PYTHON
{
    from udfio import as_tensor_table
    import pickle
    nargs = len(_columns)
    shape = tuple([_columns['arg1'][0]] + [_columns[f"arg{i}"] for i in range(2, nargs+1)])
    val = numpy.zeros(shape)
    return as_tensor_table(val)
};
SELECT * from zeros1(4);

-- zeros2
CREATE OR REPLACE
FUNCTION
zeros2(*)
RETURNS
TABLE(dim0 int, dim1 int, val float)
LANGUAGE PYTHON
{
    from udfio import as_tensor_table
    import pickle
    nargs = len(_columns)
    shape = tuple([_columns['arg1'][0]] + [_columns[f"arg{i}"] for i in range(2, nargs+1)])
    val = numpy.zeros(shape)
    return as_tensor_table(val)
};
SELECT * from zeros2(4, 5);

-- zeros3
CREATE OR REPLACE
FUNCTION
zeros3(*)
RETURNS
TABLE(dim0 int, dim1 int, dim2 int, val float)
LANGUAGE PYTHON
{
    from udfio import as_tensor_table
    import pickle
    nargs = len(_columns)
    shape = tuple([_columns['arg1'][0]] + [_columns[f"arg{i}"] for i in range(2, nargs+1)])
    val = numpy.zeros(shape)
    return as_tensor_table(val)
};
SELECT * from zeros3(4, 5, 7);

-- randints
CREATE OR REPLACE
FUNCTION
randint(*)
RETURNS
TABLE(dim0 int, dim1 int, dim2 int, val float)
LANGUAGE PYTHON
{
    from udfio import as_tensor_table
    import pickle
    nargs = len(_columns)
    low = arg1
    high = arg2
    shape = tuple([_columns[f"arg{i}"] for i in range(3, nargs+1)])
    val = numpy.random.randint(low, high, shape)
    return as_tensor_table(val)
};

SELECT * FROM randint(0,10,3, 4, 6);


