from inspect import cleandoc
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from exareme2 import DType
from exareme2.algorithms.exareme2.longitudinal_transformer import (
    LongitudinalTransformer,
)
from exareme2.algorithms.exareme2.longitudinal_transformer import (
    LongitudinalTransformerUdf,
)
from exareme2.worker_communication import BadUserInput


class TestLongitudinalTransformerUdf:
    def test_get_exec_stmt__first(self):
        kwargs = self._make_kwargs({"var": "first"})
        transf = LongitudinalTransformerUdf(flowkwargs=kwargs)

        result = transf.get_exec_stmt(udf_name=None, output_table_names=["result"])

        expected = """
        INSERT INTO result
        SELECT
            t1."row_id",
            t1."var"
        FROM
            (SELECT
                *
            FROM
                test_table
            WHERE
                "visitid"='BL') AS t1
            INNER JOIN
            (SELECT
                *
            FROM
                test_table
            WHERE
                "visitid"='FL1') AS t2
            ON t1.subjectid=t2.subjectid;"""
        assert result == cleandoc(expected)

    def test_get_exec_stmt__second(self):
        kwargs = self._make_kwargs({"var": "second"})
        transf = LongitudinalTransformerUdf(flowkwargs=kwargs)

        result = transf.get_exec_stmt(udf_name=None, output_table_names=["result"])

        expected = """
        INSERT INTO result
        SELECT
            t1."row_id",
            t2."var"
        FROM
            (SELECT
                *
            FROM
                test_table
            WHERE
                "visitid"='BL') AS t1
            INNER JOIN
            (SELECT
                *
            FROM
                test_table
            WHERE
                "visitid"='FL1') AS t2
            ON t1.subjectid=t2.subjectid;"""
        assert result == cleandoc(expected)

    def test_get_exec_stmt__diff(self):
        kwargs = self._make_kwargs({"var": "diff"})
        transf = LongitudinalTransformerUdf(flowkwargs=kwargs)

        result = transf.get_exec_stmt(udf_name=None, output_table_names=["result"])

        expected = """
        INSERT INTO result
        SELECT
            t1."row_id",
            t2."var" - t1."var"
        FROM
            (SELECT
                *
            FROM
                test_table
            WHERE
                "visitid"='BL') AS t1
            INNER JOIN
            (SELECT
                *
            FROM
                test_table
            WHERE
                "visitid"='FL1') AS t2
            ON t1.subjectid=t2.subjectid;"""
        assert result == cleandoc(expected)

    def test_get_exec_stmt__multiple_vars(self):
        kwargs = self._make_kwargs({"var1": "first", "var2": "second", "var3": "diff"})
        transf = LongitudinalTransformerUdf(flowkwargs=kwargs)

        result = transf.get_exec_stmt(udf_name=None, output_table_names=["result"])

        expected = """
        INSERT INTO result
        SELECT
            t1."row_id",
            t1."var1",
            t2."var2",
            t2."var3" - t1."var3"
        FROM
            (SELECT
                *
            FROM
                test_table
            WHERE
                "visitid"='BL') AS t1
            INNER JOIN
            (SELECT
                *
            FROM
                test_table
            WHERE
                "visitid"='FL1') AS t2
            ON t1.subjectid=t2.subjectid;"""
        assert result == cleandoc(expected)

    def test_get_exec_stmt__other_visits(self):
        kwargs = self._make_kwargs({"var": "first"}, visit1="FL1", visit2="FL2")
        transf = LongitudinalTransformerUdf(flowkwargs=kwargs)

        result = transf.get_exec_stmt(udf_name=None, output_table_names=["result"])

        expected = """
        INSERT INTO result
        SELECT
            t1."row_id",
            t1."var"
        FROM
            (SELECT
                *
            FROM
                test_table
            WHERE
                "visitid"='FL1') AS t1
            INNER JOIN
            (SELECT
                *
            FROM
                test_table
            WHERE
                "visitid"='FL2') AS t2
            ON t1.subjectid=t2.subjectid;"""
        assert result == cleandoc(expected)

    def test_output_schema(self):
        kwargs = self._make_kwargs(strategies={})
        schema = [("var1", DType.INT), ("var2", DType.FLOAT), ("var3", DType.STR)]
        transf = LongitudinalTransformerUdf(flowkwargs=kwargs, output_schema=schema)

        result = transf.output_schema

        assert result == schema

    def test_get_results(self):
        kwargs = self._make_kwargs(strategies={})
        schema = [("var1", DType.INT), ("var2", DType.FLOAT), ("var3", DType.STR)]
        transf = LongitudinalTransformerUdf(flowkwargs=kwargs, output_schema=schema)

        results = transf.get_results(["result"])

        assert len(results) == 1
        [result] = results
        assert result.table_name == "result"
        assert result.table_schema == schema
        exp_query = 'CREATE TABLE result("var1" INT,"var2" DOUBLE,"var3" VARCHAR(500));'
        assert result.create_query == exp_query

    @staticmethod
    def _make_kwargs(strategies, visit1="BL", visit2="FL1"):
        dataframe = SimpleNamespace(name="test_table")
        return {
            "dataframe": dataframe,
            "visit1": visit1,
            "visit2": visit2,
            "strategies": strategies,
        }


# Alias globalworker_db_cursor to db
@pytest.fixture(scope="module")
def db(globalworker_db_cursor):
    return globalworker_db_cursor


@pytest.mark.slow
@pytest.mark.database
@pytest.mark.usefixtures("monetdb_globalworker")
class TestLongitudinalTransformerUdf_WithDb:
    test_table = "test_longitudinal_table"

    @pytest.mark.usefixtures("delete_result_table")
    def test_create_result_table(self, db):
        output_schema = [("a", DType.INT)]
        transf = LongitudinalTransformerUdf(flowkwargs={}, output_schema=output_schema)

        udf_result, *_ = transf.get_results(output_table_names=["result"])
        create_query = udf_result.create_query
        db.execute(create_query)

        result = self._get_result(db)
        assert result == []

    @pytest.mark.usefixtures("longitudinal_dataframe", "result_table")
    def test_longitudinal_transform__no_rows(self, db):
        kwargs = self._make_kwargs(
            strategies={"numvar": "diff", "nomvar": "first"},
            visit1="BL",
            visit2="FL3",
        )
        schema = [("row_id", DType.INT), ("numvar", DType.INT), ("nomvar", DType.STR)]
        transf = LongitudinalTransformerUdf(flowkwargs=kwargs, output_schema=schema)

        exec_stmt = transf.get_exec_stmt(udf_name=None, output_table_names=["result"])
        db.execute(exec_stmt)

        result = self._get_result(db)
        assert result == []

    @pytest.mark.usefixtures("longitudinal_dataframe", "result_table")
    def test_longitudinal_transform__one_row(self, db):
        kwargs = self._make_kwargs(
            strategies={"numvar": "diff", "nomvar": "first"},
            visit1="BL",
            visit2="FL2",
        )
        schema = [("row_id", DType.INT), ("numvar", DType.INT), ("nomvar", DType.STR)]
        transf = LongitudinalTransformerUdf(flowkwargs=kwargs, output_schema=schema)

        exec_stmt = transf.get_exec_stmt(udf_name=None, output_table_names=["result"])
        db.execute(exec_stmt)

        result = self._get_result(db)
        assert result == [(0, 2, "a")]

    @pytest.mark.usefixtures("longitudinal_dataframe", "result_table")
    def test_longitudinal_transform__multiple_rows(self, db):
        kwargs = self._make_kwargs(
            strategies={"numvar": "diff", "nomvar": "first"},
            visit1="BL",
            visit2="FL1",
        )
        schema = [("row_id", DType.INT), ("numvar", DType.INT), ("nomvar", DType.STR)]
        transf = LongitudinalTransformerUdf(flowkwargs=kwargs, output_schema=schema)

        exec_stmt = transf.get_exec_stmt(udf_name=None, output_table_names=["result"])
        db.execute(exec_stmt)

        result = self._get_result(db)
        assert result == [(0, 1, "a"), (3, 2, "a")]

    @pytest.mark.usefixtures("longitudinal_dataframe", "result_table")
    def test_longitudinal_transform__second_strategy(self, db):
        kwargs = self._make_kwargs(
            strategies={"numvar": "diff", "nomvar": "second"},
            visit1="BL",
            visit2="FL2",
        )
        schema = [("row_id", DType.INT), ("numvar", DType.INT), ("nomvar", DType.STR)]
        transf = LongitudinalTransformerUdf(flowkwargs=kwargs, output_schema=schema)

        exec_stmt = transf.get_exec_stmt(udf_name=None, output_table_names=["result"])
        db.execute(exec_stmt)

        result = self._get_result(db)
        assert result == [(0, 2, "b")]

    @pytest.fixture(scope="class")
    def longitudinal_dataframe(self, db):
        self._create_longitudinal_table(db)
        self._populate_longitudinal_table(db)
        try:
            yield
        finally:
            self._delete_longitudinal_table(db)

    @pytest.fixture(scope="function")
    def result_table(self, db, delete_result_table):
        db.execute("CREATE TABLE result(row_id INT, numvar INT, nomvar TEXT)")
        yield

    @pytest.fixture(scope="function")
    def delete_result_table(self, db):
        try:
            yield
        finally:
            db.execute("DROP TABLE IF EXISTS result")

    def _create_longitudinal_table(self, db):
        sql = f"""
        CREATE TABLE {self.test_table}(row_id INT,
                                       subjectid TEXT,
                                       visitid TEXT,
                                       numvar INT,
                                       nomvar TEXT);
        """
        db.execute(sql)

    def _populate_longitudinal_table(self, db):
        sql = f"""
        INSERT INTO {self.test_table} VALUES
            (0 , 1 , 'BL'  , 1 , 'a'),
            (1 , 1 , 'FL1' , 2 , 'b'),
            (2 , 1 , 'FL2' , 3 , 'b'),
            (3 , 2 , 'BL'  , 2 , 'a'),
            (4 , 2 , 'FL1' , 4 , 'b');
        """
        db.execute(sql)

    def _delete_longitudinal_table(self, db):
        db.execute(f"DROP TABLE {self.test_table};")

    @staticmethod
    def _get_result(db):
        return db.execute("SELECT * FROM result").fetchall()

    def _make_kwargs(self, strategies, visit1, visit2):
        dataframe = SimpleNamespace(name=self.test_table)
        return {
            "dataframe": dataframe,
            "visit1": visit1,
            "visit2": visit2,
            "strategies": strategies,
        }


class TestLongitudinalTransformer:
    def test_transform_schema__valid(self):
        metadata = {
            "numvar": {"sql_type": "int", "is_categorical": False},
            "nomvar": {"sql_type": "text", "is_categorical": True},
        }
        engine = Mock()
        strategies = {"numvar": "diff", "nomvar": "first"}
        transf = LongitudinalTransformer(
            engine, metadata, strategies, visit1="BL", visit2="FL1"
        )

        transf.transform(X=None)

        expected_schema = [
            ("row_id", DType.INT),
            ("numvar_diff", DType.INT),
            ("nomvar", DType.STR),
        ]
        call_kwargs = engine.run_udf_on_local_workers.call_args.kwargs
        assert call_kwargs["output_schema"] == expected_schema

    def test_transform_schema__invalid_diff(self):
        metadata = {"nomvar": {"sql_type": "text", "is_categorical": True}}
        engine = Mock()
        strategies = {"nomvar": "diff"}

        with pytest.raises(BadUserInput):
            LongitudinalTransformer(
                engine, metadata, strategies, visit1="BL", visit2="FL1"
            )
