import json

from mipengine.exceptions import BadUserInput
from mipengine.udfgen import DEFERRED
from mipengine.udfgen import literal
from mipengine.udfgen import relation
from mipengine.udfgen import state
from mipengine.udfgen import transfer
from mipengine.udfgen import udf


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
                output_schema=X.full_schema.to_list(),
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
                output_schema=X.full_schema.to_list(),
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
                output_schema=y.full_schema.to_list(),
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
                output_schema=y.full_schema.to_list(),
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
