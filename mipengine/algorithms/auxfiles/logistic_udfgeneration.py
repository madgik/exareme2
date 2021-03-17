from mipengine.node.udfgen import TableInfo
from mipengine.node.udfgen import ColumnInfo
from mipengine.node.udfgen import generate_udf_application_queries
import mipengine.algorithms.logistic_regression

X = TableInfo(name="features", schema=[ColumnInfo("feat1", "float"), ColumnInfo("feat2", "float")])
y = TableInfo(name="target", schema=[ColumnInfo("target", "int")])
tens1 = TableInfo(name="tens1", schema=[ColumnInfo("dim0", "int"), ColumnInfo("val", "float")])
tens2 = TableInfo(name="tens2", schema=[ColumnInfo("dim0", "int"), ColumnInfo("val", "float")])
matr = TableInfo(name="matrix", schema=[ColumnInfo("dim0", "int"), ColumnInfo("dim1", "int"), ColumnInfo("val", "float")])
vec = TableInfo(name="vector", schema=[ColumnInfo("dim0", "int"), ColumnInfo("val", "float")])
diag = TableInfo(name="diag", schema=[ColumnInfo("dim0", "int"), ColumnInfo("val", "float")])

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

DROP TABLE IF EXISTS merge_table;
CREATE TABLE merge_table(node_id int, dim0 INT, dim1 INT, val FLOAT);
INSERT INTO merge_table VALUES (0, 0, 0, 0.50);
INSERT INTO merge_table VALUES (0, 0, 1, 0.73);
INSERT INTO merge_table VALUES (0, 1, 0, 0.93);
INSERT INTO merge_table VALUES (0, 1, 1, 0.111);
INSERT INTO merge_table VALUES (1, 0, 0, 2.50);
INSERT INTO merge_table VALUES (1, 0, 1, 2.73);
INSERT INTO merge_table VALUES (1, 1, 0, 2.93);
INSERT INTO merge_table VALUES (1, 1, 1, 2.111);
INSERT INTO merge_table VALUES (2, 0, 0, 2.50);
INSERT INTO merge_table VALUES (2, 0, 1, 2.73);
INSERT INTO merge_table VALUES (2, 1, 0, 2.93);
INSERT INTO merge_table VALUES (2, 1, 1, 2.111);
INSERT INTO merge_table VALUES (3, 0, 0, 2.50);
INSERT INTO merge_table VALUES (3, 0, 1, 2.73);
INSERT INTO merge_table VALUES (3, 1, 0, 2.93);
INSERT INTO merge_table VALUES (3, 1, 1, 2.111);
"""

udf, query = generate_udf_application_queries("logistic_regression.relation_to_vector", [y], {})
print(udf.substitute(udf_name="relation_to_vector"))
print(query.substitute(udf_name="relation_to_vector", table_name="relation_to_vector_results", node_id='12345'))
print()

udf, query = generate_udf_application_queries("logistic_regression.relation_to_matrix", [X], {})
print(udf.substitute(udf_name="relation_to_matrix"))
print(query.substitute(udf_name="relation_to_matrix", table_name="relation_to_matrix_result", node_id='12345'))
print()

udf, query = generate_udf_application_queries("sql.zeros1", [5], {})
print(udf.substitute(udf_name="zeros1"))
print(query.substitute(udf_name="zeros1", table_name="zeros1_result", node_id='12345'))
print()
# UDF result
# +------+--------------------------+
# | dim0 | val                      |
# +======+==========================+
# |    0 |                        0 |
# |    1 |                        0 |
# |    2 |                        0 |
# |    3 |                        0 |
# |    4 |                        0 |
# +------+--------------------------+

udf, query = generate_udf_application_queries("sql.matrix_dot_vector", [matr, vec], {})
print(udf.substitute(udf_name="matrix_dot_vector"))
print(query.substitute(udf_name="matrix_dot_vector", table_name="matrix_dot_vector_results", node_id='12345'))
print()
# UDF result
# +------+--------------------------+
# | dim0 | val                      |
# +======+==========================+
# |    0 |                   69.284 |
# |    1 |                   77.074 |
# +------+--------------------------+

udf, query = generate_udf_application_queries("sql.tensor1_mult", [tens1, tens1], {})
print(udf.substitute(udf_name="tensor_mult"))
print(query.substitute(udf_name="tensor_mult", table_name="tensor_mult_result", node_id='12345'))
print()
# UDF result
# +------+--------------------------+
# | dim0 | val                      |
# +======+==========================+
# |    0 |                     0.25 |
# |    1 |       0.5328999999999999 |
# |    2 |       0.8649000000000001 |
# |    3 |                 0.012321 |
# +------+--------------------------+

udf, query = generate_udf_application_queries("sql.tensor1_add", [tens1, tens1], {})
print(udf.substitute(udf_name="tensor_add"))
print(query.substitute(udf_name="tensor_add", table_name="tensor_add_result", node_id='12345'))
print()
# UDF result
# +------+--------------------------+
# | dim0 | val                      |
# +======+==========================+
# |    0 |                        1 |
# |    1 |                     1.46 |
# |    2 |                     1.86 |
# |    3 |                    0.222 |
# +------+--------------------------+

udf, query = generate_udf_application_queries("sql.tensor1_sub", [tens1, tens1], {})
print(udf.substitute(udf_name="tensor_sub"))
print(query.substitute(udf_name="tensor_sub", table_name="tensor_sub_result", node_id='12345'))
print()
# UDF result
# +------+--------------------------+
# | dim0 | val                      |
# +======+==========================+
# |    0 |                        0 |
# |    1 |                        0 |
# |    2 |                        0 |
# |    3 |                        0 |
# +------+--------------------------+

udf, query = generate_udf_application_queries("sql.tensor1_div", [tens1, tens1], {})
print(udf.substitute(udf_name="tensor_div"))
print(query.substitute(udf_name="tensor_div", table_name="tensor_div_result", node_id='12345'))
print()
# UDF result
# +------+--------------------------+
# | dim0 | val                      |
# +======+==========================+
# |    0 |                        1 |
# |    1 |                        1 |
# |    2 |                        1 |
# |    3 |                        1 |
# +------+--------------------------+

udf, query = generate_udf_application_queries("sql.const_tensor1_sub", [1, tens1], {})
print(udf.substitute(udf_name="const_tensor_sub1"))
print(query.substitute(udf_name="const_tensor_sub1", table_name="const_tensor_sub1_result", node_id='12345'))
print()
# UDF result
# +------+--------------------------+
# | dim0 | v                        |
# +======+==========================+
# |    0 |                      0.5 |
# |    1 |                     0.27 |
# |    2 |      0.06999999999999995 |
# |    3 |                    0.889 |
# +------+--------------------------+

udf, query = generate_udf_application_queries("sql.mat_transp_dot_diag_dot_mat", [matr, vec], {})
print(udf.substitute(udf_name="mat_transp_dot_diag_dot_mat"))
print(query.substitute(udf_name="mat_transp_dot_diag_dot_mat", table_name="mat_transp_dot_diag_dot_mat_result", node_id='12345'))
print()
# UDF result
# +------+------+--------------------------+
# | dim0 | dim1 | val                      |
# +======+======+==========================+
# |    0 |    0 |                 834.9442 |
# |    0 |    1 |                 674.2284 |
# |    1 |    0 |                 674.2284 |
# |    1 |    1 |                 611.3466 |
# +------+------+--------------------------+

udf, query = generate_udf_application_queries("sql.mat_transp_dot_diag_dot_vec", [matr, diag, vec], {})
print(udf.substitute(udf_name="mat_transp_dot_diag_dot_vec"))
print(query.substitute(udf_name="mat_transp_dot_diag_dot_vec", table_name="mat_transp_dot_diag_dot_vec_result", node_id='12345'))
print()
# UDF result
# +------+--------------------------+
# | dim0 | val                      |
# +======+==========================+
# |    0 |                285.25462 |
# |    1 |       225.81623999999996 |
# +------+--------------------------+

udf, query = generate_udf_application_queries("logistic_regression.tensor_expit", [tens1], {})
print(udf.substitute(udf_name="tensor_expit"))
print(query.substitute(udf_name="tensor_expit", table_name="expit_result", node_id='12345'))
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
print(query.substitute(udf_name="mat_inverse", table_name="mat_inverse_result", node_id='12345'))
print()
# UDF result
# +------+------+--------------------------+
# | dim0 | dim1 | val                      |
# +======+======+==========================+
# |    0 |    0 |     -0.13379583746283444 |
# |    0 |    1 |      0.15361744301288402 |
# |    1 |    0 |       0.1866534522629666 |
# |    1 |    1 |      -0.0908490254377271 |
# +------+------+--------------------------+

merge_table = TableInfo(name="merge_table", schema=[ColumnInfo("node_id", "int"), ColumnInfo("dim0", "int"),ColumnInfo("dim1", "int"), ColumnInfo("val", "float")])
udf, query = generate_udf_application_queries("reduce.sum_tensors", [merge_table], {})
print(udf.substitute(udf_name="sum_tensors"))
print(query.substitute(udf_name="sum_tensors", table_name="reduced_result", node_id='12345'))
print()
# UDF result
# +------+------+--------------------------+
# | dim0 | dim1 | val                      |
# +======+======+==========================+
# |    0 |    0 |                        8 |
# |    0 |    1 |                     8.92 |
# |    1 |    0 |                     9.72 |
# |    1 |    1 |        6.444000000000001 |
# +------+------+--------------------------+
