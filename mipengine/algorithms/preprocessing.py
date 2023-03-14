import json

from mipengine import DType
from mipengine.algorithms.helpers import get_transfer_data
from mipengine.exceptions import BadUserInput
from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.udfgen import DEFERRED
from mipengine.udfgen import literal
from mipengine.udfgen import merge_transfer
from mipengine.udfgen import relation
from mipengine.udfgen import state
from mipengine.udfgen import tensor
from mipengine.udfgen import transfer
from mipengine.udfgen import udf


# TODO rewrite DummyEncoder using FormulaTransformer
class DummyEncoder:
    def __init__(self, engine, variables, metadata, intercept=True):
        self._local_run = engine.run_udf_on_local_nodes
        self._global_run = engine.run_udf_on_global_node
        self._categorical_vars = [
            varname for varname in variables.x if metadata[varname]["is_categorical"]
        ]
        self._numerical_vars = [
            varname
            for varname in variables.x
            if not metadata[varname]["is_categorical"]
        ]
        self.intercept = intercept
        self.new_varnames = None

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
            func="create_dummy_encoded_design_matrix",
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

    def transform(self, x):
        enums = self._gather_enums(x)
        self.new_varnames = self._get_new_variable_names(self._numerical_vars, enums)
        if self._categorical_vars or self.intercept:
            x = self._create_design_matrix(x, enums)
        return x


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


class KFold:
    """Slits dataset into train and test sets for performing k-flod cross-validation

    NOTE: This is currently implemented in a very inefficient maner, making one
    `run_udf_on_local_nodes` per split, per table. The reason is limitations in
    the current UDF generator. In the future this class might be re-implemented
    more efficiently. However, the interface won't change.
    """

    def __init__(self, engine, n_splits):
        """
        Parameters
        ----------
        angine: AlgorithmExecutionEngine
        n_splits: int
        """
        self._local_run = engine.run_udf_on_local_nodes
        self._global_run = engine.run_udf_on_global_node
        self.n_splits = n_splits

    def split(self, X, y):
        local_state, local_condition_transfers = self._local_run(
            func=self._split_local,
            keyword_args={"x": X, "y": y, "n_splits": self.n_splits},
            share_to_global=[False, True],
        )

        [transfer_data] = local_condition_transfers.get_table_data()
        conditions = [json.loads(t) for t in transfer_data]
        if not all(cond["n_obs >= n_splits"] for cond in conditions):
            raise BadUserInput(
                "Cross validation cannot run because some of the nodes "
                "participating in the experiment have a number of observations "
                f"smaller than the number of splits, {self.n_splits}."
            )

        x_train = [
            self._local_run(
                func=self._get_split_local,
                keyword_args=dict(
                    local_state=local_state,
                    i=i,
                    key="x_train",
                ),
                share_to_global=[False],
                output_schema=X.full_schema,
            )
            for i in range(self.n_splits)
        ]

        x_test = [
            self._local_run(
                func=self._get_split_local,
                keyword_args=dict(
                    local_state=local_state,
                    i=i,
                    key="x_test",
                ),
                share_to_global=[False],
                output_schema=X.full_schema,
            )
            for i in range(self.n_splits)
        ]

        y_train = [
            self._local_run(
                func=self._get_split_local,
                keyword_args=dict(
                    local_state=local_state,
                    i=i,
                    key="y_train",
                ),
                share_to_global=[False],
                output_schema=y.full_schema,
            )
            for i in range(self.n_splits)
        ]

        y_test = [
            self._local_run(
                func=self._get_split_local,
                keyword_args=dict(
                    local_state=local_state,
                    i=i,
                    key="y_test",
                ),
                share_to_global=[False],
                output_schema=y.full_schema,
            )
            for i in range(self.n_splits)
        ]

        return x_train, x_test, y_train, y_test

    @staticmethod
    @udf(
        x=relation(),
        y=relation(),
        n_splits=literal(),
        return_type=[state(), transfer()],
    )
    def _split_local(x, y, n_splits):
        import itertools

        import sklearn.model_selection

        # Error handling within a UDF is not possible. Instead, I evaluate the
        # necessary condition len(y) >= n_splits, and proceed with the
        # computation accordingly. Moreover, the condition value is sent to the
        # algorithm flow, where it is handled, using an auxiliary transfer
        # object.
        n_obs = len(y)
        if n_obs >= n_splits:
            kf = sklearn.model_selection.KFold(n_splits=n_splits)

            x_cv_indices, y_cv_indices = itertools.tee(kf.split(x), 2)

            x_train, x_test = [], []
            for train_idx, test_idx in x_cv_indices:
                x_train.append(x.iloc[train_idx])
                x_test.append(x.iloc[test_idx])

            y_train, y_test = [], []
            for train_idx, test_idx in y_cv_indices:
                y_train.append(y.iloc[train_idx])
                y_test.append(y.iloc[test_idx])

            state_ = dict(
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
            )
        else:
            state_ = {}
        transfer_ = {"n_obs >= n_splits": n_obs >= n_splits}
        return state_, transfer_

    @staticmethod
    @udf(
        local_state=state(),
        i=literal(),
        key=literal(),
        return_type=relation(schema=DEFERRED),
    )
    def _get_split_local(local_state, i, key):
        split = local_state[key][i]
        result = split
        return result


# TODO extract CategoriesFinder class
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
        columns = [ColumnInfo(name=name, dtype=DType.INT) for name in column_names]
        if X.index:
            columns.insert(0, ColumnInfo(name=X.index, dtype=DType.INT))
        return TableSchema(columns=columns)

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
