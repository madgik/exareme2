import json

from mipengine.udfgen import DEFERRED
from mipengine.udfgen import literal
from mipengine.udfgen import merge_transfer
from mipengine.udfgen import relation
from mipengine.udfgen import state
from mipengine.udfgen import tensor
from mipengine.udfgen import transfer
from mipengine.udfgen import udf


class DummyEncoder:
    def __init__(self, executor, intercept=True):
        self._local_run = executor.run_udf_on_local_nodes
        self._global_run = executor.run_udf_on_global_node
        self._categorical_vars = [
            varname
            for varname in executor.x_variables
            if executor.metadata[varname]["is_categorical"]
        ]
        self._numerical_vars = [
            varname
            for varname in executor.x_variables
            if not executor.metadata[varname]["is_categorical"]
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
                share_to_locals=[True],
            )
            enums = json.loads(global_transfer.get_table_data()[1][0])
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


def relation_to_vector(rel, executor):
    return executor.run_udf_on_local_nodes(
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

    def __init__(self, executor, n_splits):
        """
        Parameters
        ----------
        executor: _AlgorithmExecutionInterface
        n_splits: int
        """
        self._local_run = executor.run_udf_on_local_nodes
        self._global_run = executor.run_udf_on_global_node
        self.n_splits = n_splits

    def split(self, X, y):
        local_state = self._local_run(
            func=self._split_local,
            keyword_args={"x": X, "y": y, "n_splits": self.n_splits},
            share_to_global=[False],
        )

        x_return_schema = X.get_table_schema()
        y_return_schema = y.get_table_schema()

        x_train = [
            self._local_run(
                func=self._get_split_local,
                keyword_args=dict(
                    local_state=local_state,
                    i=i,
                    key="x_train",
                ),
                share_to_global=[False],
                output_schema=x_return_schema,
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
                output_schema=x_return_schema,
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
                output_schema=y_return_schema,
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
                output_schema=y_return_schema,
            )
            for i in range(self.n_splits)
        ]

        return x_train, x_test, y_train, y_test

    @staticmethod
    @udf(x=relation(), y=relation(), n_splits=literal(), return_type=state())
    def _split_local(x, y, n_splits):
        import itertools

        import sklearn.model_selection

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

        state_ = dict(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
        return state_

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
