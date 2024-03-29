from typing import List

from exareme2 import DType
from exareme2.algorithms.exareme2.helpers import get_transfer_data
from exareme2.algorithms.exareme2.udfgen import DEFERRED
from exareme2.algorithms.exareme2.udfgen import AdhocUdfGenerator
from exareme2.algorithms.exareme2.udfgen import literal
from exareme2.algorithms.exareme2.udfgen import merge_transfer
from exareme2.algorithms.exareme2.udfgen import relation
from exareme2.algorithms.exareme2.udfgen import tensor
from exareme2.algorithms.exareme2.udfgen import transfer
from exareme2.algorithms.exareme2.udfgen import udf
from exareme2.algorithms.exareme2.udfgen.udfgen_DTOs import UDFGenTableResult


# TODO extract EnumAggregator class
class DummyEncoder:
    def __init__(self, engine, metadata, intercept=True):
        self._local_run = engine.run_udf_on_local_nodes
        self._global_run = engine.run_udf_on_global_node
        self.metadata = metadata
        self.intercept = intercept
        self.new_varnames = None
        self._categorical_vars = None
        self._numerical_vars = None

    def _gather_enums(self, x):
        if self._categorical_vars:
            local_transfers = self._local_run(
                func=self._gather_enums_local,
                keyword_args={"x": x, "categorical_vars": self._categorical_vars},
                share_to_global=[True],
            )
            global_transfer = self._global_run(
                func=self._gather_enums_global,
                keyword_args=dict(local_transfers=local_transfers),
            )
            enums = get_transfer_data(global_transfer)
            enums = {varname: sorted(e)[1:] for varname, e in enums.items()}
            enums = {
                varname: [
                    {
                        "code": code,
                        "dummy": f"{varname}__{i}",
                        "label": f"{varname}[{code}]",
                    }
                    for i, code in enumerate(e)
                ]
                for varname, e in enums.items()
            }
            return enums
        return {}

    @staticmethod
    @udf(x=relation(), categorical_vars=literal(), return_type=transfer())
    def _gather_enums_local(x, categorical_vars):
        categorical_vars = [varname for varname in categorical_vars]
        enumerations = {}
        for cat in categorical_vars:
            enumerations[cat] = list(x[cat].unique())

        transfer_ = dict(enumerations=enumerations)
        return transfer_

    @staticmethod
    @udf(local_transfers=merge_transfer(), return_type=transfer())
    def _gather_enums_global(local_transfers):
        from functools import reduce

        def reduce_enums(varname):
            return list(
                reduce(
                    lambda a, b: set(a) | set(b),
                    [loctrans["enumerations"][varname] for loctrans in local_transfers],
                    {},
                )
            )

        keys = local_transfers[0]["enumerations"].keys()
        enumerations = {key: reduce_enums(key) for key in keys}

        return enumerations

    def _create_design_matrix(self, x, enums):
        design_matrix = self._local_run(
            func=DummyEncoderUdf,
            keyword_args=dict(
                x=x,
                enums=enums,
                numerical_vars=self._numerical_vars,
                intercept=self.intercept,
            ),
            share_to_global=[False],
        )
        return design_matrix

    def _get_new_variable_names(self, numerical_vars, enums):
        names = []
        if self.intercept:
            names.append("Intercept")
        names.extend([e["label"] for enum in enums.values() for e in enum])
        names.extend([varname for varname in numerical_vars])
        return names

    def _split_variables(self, x):
        variables = x.columns
        metadata = self.metadata
        self._categorical_vars = [
            varname for varname in variables if metadata[varname]["is_categorical"]
        ]
        self._numerical_vars = [
            varname for varname in variables if not metadata[varname]["is_categorical"]
        ]

    def transform(self, x):
        self._split_variables(x)
        enums = self._gather_enums(x)
        self.new_varnames = self._get_new_variable_names(self._numerical_vars, enums)
        if self._categorical_vars or self.intercept:
            x = self._create_design_matrix(x, enums)
        return x


class DummyEncoderUdf(AdhocUdfGenerator):
    @property
    def output_schema(self):
        columns = [("row_id", DType.INT)]
        if self.intercept:
            columns.append(("intercept", DType.FLOAT))

        enums = [enum for enums in self.enums.values() for enum in enums]
        dummy_columns = [enum["dummy"] for enum in enums]
        columns.extend([(col, DType.FLOAT) for col in dummy_columns])

        columns.extend([(col, DType.FLOAT) for col in self.numerical_vars])

        return columns

    def get_definition(self, udf_name: str, output_table_names: List[str]) -> str:
        return ""

    def get_exec_stmt(self, udf_name: None, output_table_names: List[str]) -> str:
        main_output_name, *_ = output_table_names

        columns = [self.ast.Column(name="row_id")]
        if self.intercept:
            columns.append(self.ast.ConstColumn(value=1, alias="intercept"))

        dummy_columns = [
            self.ast.ConstColumn(
                value=f"CASE WHEN {varname} = '{enum['code']}' THEN 1 ELSE 0 END",
                alias=enum["dummy"],
            )
            for varname in self.enums.keys()
            for enum in self.enums[varname]
        ]
        columns.extend(dummy_columns)

        num_columns = [self.ast.Column(name=varname) for varname in self.numerical_vars]
        columns.extend(num_columns)

        table = self.ast.Table(name=self.x.name, columns=columns)
        sel = self.ast.Select(columns=table.columns, from_=[table])
        return self.ast.Insert(table=main_output_name, values=sel).compile()

    def get_results(self, output_table_names: List[str]) -> List[UDFGenTableResult]:
        main_output_name, *_ = output_table_names
        create = self.ast.CreateTable(main_output_name, self.output_schema).compile()
        return [
            UDFGenTableResult(
                table_name=main_output_name,
                table_schema=self.output_schema,
                create_query=create,
            )
        ]


class LabelBinarizer:
    """Transforms a table of labels to binary in one-vs-all fashion

    This is used in classification models where the input target table is
    composed of not necessarily numeric and not necessarily binary elements. It
    is a port of `sklearn.preprocessing.LabelBinarizer` with one important
    difference.

    The original discovers all the classes present in the data, during the
    `fit` part, and returns a matrix where each column corresponds to one class
    being considered 'positive'.

    In our case this might lead to errors in cases where not all classes are
    present in every node, thus causing mismatches between local results. The
    remedy is to provide a `positive_class` parameter and to return only the
    corresponding binary column.

    Attributes
    ----------
    positive_class : str
        The class that should be considered positive, while all others are
        considered negative.
    """

    def __init__(self, engine, positive_class):
        self._local_run = engine.run_udf_on_local_nodes
        self._global_run = engine.run_udf_on_global_node
        self.positive_class = positive_class

    def transform(self, y):
        return self._local_run(
            self._transform_local,
            keyword_args={"y": y, "positive_class": self.positive_class},
        )

    @staticmethod
    @udf(
        y=relation(),
        positive_class=literal(),
        return_type=relation(schema=[("row_id", int), ("ybin", int)]),
    )
    def _transform_local(y, positive_class):
        import pandas as pd

        ybin = y == positive_class
        result = pd.DataFrame({"ybin": ybin.to_numpy().reshape((-1,))}, index=y.index)
        return result


def relation_to_vector(rel, engine):
    return engine.run_udf_on_local_nodes(
        func=relation_to_vector_local_udf,
        keyword_args={"rel": rel},
        share_to_global=[False],
    )


@udf(rel=relation(), return_type=tensor(dtype=float, ndims=1))
def relation_to_vector_local_udf(rel):
    return rel


# TODO extract EnumAggregator class
class FormulaTransformer:
    """Transforms a table based on R style model formula.

    Internally this class uses the patsy library to implement formulas.
    Documentation on how formulas work can be found in https://patsy.readthedocs.io/
    """

    def __init__(self, engine, variables, metadata, formula):
        """
        Parameters
        ----------
        engine : AlgorithmExecutionEngine
            Instance of algorithm execution engine.
        formula : str
            R style model formula.
        """
        self._local_run = engine.run_udf_on_local_nodes
        self._global_run = engine.run_udf_on_global_node
        self._categorical_vars = [
            varname for varname in variables.x if metadata[varname]["is_categorical"]
        ]
        self._formula = formula

    def transform(self, X):
        """
        Transforms X based on model formula

        Parameters
        ----------
        X : LocalNodeTable

        Returns
        -------
        LocalNodeTable
        """
        self.enums = self._gather_enums(X)
        schema = self._compute_output_schema(X, self.enums)
        return self._local_run(
            func=self._transform_local,
            positional_args=(X, self._formula, self.enums),
            output_schema=schema,
        )

    def _gather_enums(self, x):
        """In order to compute columns corresponding to terms of the formula,
        we need to know all actual categorical enumerations present in the data.
        This method gathers enumerations from local nodes by calling one local and
        one global UDF."""
        if self._categorical_vars:
            local_transfers = self._local_run(
                func=self._gather_enums_local,
                keyword_args={"x": x, "categorical_vars": self._categorical_vars},
                share_to_global=[True],
            )
            global_transfer = self._global_run(
                func=self._gather_enums_global,
                keyword_args=dict(local_transfers=local_transfers),
            )
            enums = get_transfer_data(global_transfer)
            return enums
        return {}

    @staticmethod
    @udf(x=relation(), categorical_vars=literal(), return_type=transfer())
    def _gather_enums_local(x, categorical_vars):
        categorical_vars = [varname for varname in categorical_vars]
        enumerations = {}
        for cat in categorical_vars:
            enumerations[cat] = list(x[cat].unique())

        transfer_ = dict(enumerations=enumerations)
        return transfer_

    @staticmethod
    @udf(local_transfers=merge_transfer(), return_type=transfer())
    def _gather_enums_global(local_transfers):
        from functools import reduce

        def reduce_enums(varname):
            return list(
                reduce(
                    lambda a, b: set(a) | set(b),
                    [loctrans["enumerations"][varname] for loctrans in local_transfers],
                    {},
                )
            )

        keys = local_transfers[0]["enumerations"].keys()
        enumerations = {key: reduce_enums(key) for key in keys}

        return enumerations

    def _compute_output_schema(self, X, enums):
        """Since the formula transformation generates new columns and the resulting
        table schema must be at UDF definition time, we need to compute the output
        schema in advance. This is done by applying the same transformation to
        a dummy table, using patsy, and then looking at the produced column names."""
        column_names = self._compute_new_column_names(X.columns, enums)
        schema = [(name, DType.INT) for name in column_names]
        if X.index:
            schema.insert(0, (X.index, DType.INT))
        return schema

    def _compute_new_column_names(self, old_column_names, enums):
        import pandas as pd
        from patsy import dmatrix

        empty_data = {col: [] for col in old_column_names}
        empty_df = pd.DataFrame(data=empty_data)
        for var, categories in enums.items():
            empty_df[var] = pd.Categorical(empty_df[var], categories=categories)
        mat = dmatrix(self._formula, empty_df)
        self.design_info = mat.design_info
        new_column_names = [f"_c{i}" for i in range(len(mat.design_info.column_names))]
        return new_column_names

    @staticmethod
    @udf(
        X=relation(),
        formula=literal(),
        enums=literal(),
        return_type=relation(schema=DEFERRED),
    )
    def _transform_local(X, formula, enums):
        import pandas as pd
        from patsy import dmatrix

        for var, categories in enums.items():
            X[var] = pd.Categorical(X[var], categories=categories)

        result = dmatrix(formula, X, return_type="dataframe")
        return result
