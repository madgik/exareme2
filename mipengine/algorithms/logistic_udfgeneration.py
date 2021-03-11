from mipengine.node.udfgen import TableInfo
from mipengine.node.udfgen import ColumnInfo
from mipengine.node.udfgen import generate_udf_application_queries
import mipengine.algorithms.logistic_regression

tens1 = TableInfo(name="tens1", schema=[ColumnInfo("dim0", "int"), ColumnInfo("val", "float")])
tens2 = TableInfo(name="tens2", schema=[ColumnInfo("dim0", "int"), ColumnInfo("val", "float")])
CREATE_TABLES = """\
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
"""

udf, query = generate_udf_application_queries("logistic_regression.tensor_expit", [tens1], {})
print(udf.substitute(udf_name="tensor_expit"))
print(query.substitute(udf_name="tensor_expit", table_name="expit_result"))
print()
# UDF result
# +------+--------------------------+
# | dim0 | val                      |
# +======+==========================+
# |    0 |       0.6224593312018546 |
# |    1 |       0.6748052725823135 |
# |    2 |       0.7170752854929726 |
# |    3 |       0.5277215427491645 |
# +------+--------------------------+

udf, query = generate_udf_application_queries("logistic_regression.logistic_loss", [tens1, tens2], {})
print(udf.substitute(udf_name="logistic_loss"))
print(query.substitute(udf_name="logistic_loss", table_name="logistic_loss_result"))
print()
# UDF result
# +--------------------------+
# | v                        |
# +==========================+
# |       -5.005064700386452 |
# +--------------------------+

udf, query = generate_udf_application_queries("logistic_regression.tensor_max_abs_diff", [tens1, tens2], {})
print(udf.substitute(udf_name="tensor_max_abs_diff"))
print(query.substitute(udf_name="tensor_max_abs_diff", table_name="tensor_max_abs_diff_result"))
print()
# UDF result
# +--------------------------+
# | v                        |
# +==========================+
# |       0.8170000000000001 |
# +--------------------------+

tens3 = TableInfo(name="tens3", schema=[ColumnInfo("dim0", "int"),ColumnInfo("dim1", "int"), ColumnInfo("val", "float")])
udf, query = generate_udf_application_queries("logistic_regression.mat_inverse", [tens3], {})
print(udf.substitute(udf_name="mat_inverse"))
print(query.substitute(udf_name="mat_inverse", table_name="mat_inverse_result"))
# UDF result
# +------+------+--------------------------+
# | dim0 | dim1 | val                      |
# +======+======+==========================+
# |    0 |    0 |     -0.13379583746283444 |
# |    0 |    1 |      0.15361744301288402 |
# |    1 |    0 |       0.1866534522629666 |
# |    1 |    1 |      -0.0908490254377271 |
# +------+------+--------------------------+
