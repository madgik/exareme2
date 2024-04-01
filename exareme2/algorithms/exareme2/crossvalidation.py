import json
import typing as t

from exareme2.algorithms.exareme2.udfgen import DEFERRED
from exareme2.algorithms.exareme2.udfgen import literal
from exareme2.algorithms.exareme2.udfgen import relation
from exareme2.algorithms.exareme2.udfgen import state
from exareme2.algorithms.exareme2.udfgen import transfer
from exareme2.algorithms.exareme2.udfgen import udf
from exareme2.worker_communication import BadUserInput


class KFold:
    """Slits dataset into train and test sets for performing k-flod cross-validation

    NOTE: This is currently implemented in a very inefficient maner, making one
    `run_udf_on_local_workers` per split, per table. The reason is limitations in
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
        self._local_run = engine.run_udf_on_local_workers
        self._global_run = engine.run_udf_on_global_worker
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
                "Cross validation cannot run because some of the workers "
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


_PredictionType = t.Literal["values", "probabilities"]


def cross_validate(X, y, models, splitter, pred_type: _PredictionType):
    """
    Performs cross-validation on estimator models

    This function encapsulates the common logic found in all cross-validation
    algorithms. The function takes data (in the form of X, y), the models and a
    splitter. It then
        - Splits the data according to splitter
        - Calls `model.fit(X, y)` on every model with the train sets
        - Calls `model.predict(X)` on every model with the test set

    Parameters
    ----------
    X : LocalWorkersTable
        A table of features
    y : LocalWorkersTable
        A table of targets
    models : list of objects supporting `fit` and `predict` or `predict_proba`
        The estimator models used in cross-validation. These are mutated by the
        function since `fit` is called on each.
    splitter : object supporting `split`
        Instance of splitter object. Calling `split` should return x_train,
        y_train, x_test, y_test.
    pred_type : str, either "values" or "probabilities"
        Type of prediction. "values" returns the predicted values,
        "probabilities" returns the prediction probabilities. Typically
        "values" is used in regression and "probs" in classification.

    Returns
    -------
    List[LocalWorkersTable]
        A table of predictions for each split
    List[LocalWorkersTable]
        The testing set, i.e. a table of true values for each split
    """
    X_train, X_test, y_train, y_test = splitter.split(X, y)

    for model, X, y in zip(models, X_train, y_train):
        model.fit(X=X, y=y)

    if pred_type == "values":
        methodname = "predict"
    elif pred_type == "probabilities":
        methodname = "predict_proba"
    else:
        raise ValueError("`prediction_type` should be 'values' or 'probabilities")

    y_pred = [getattr(model, methodname)(X) for model, X in zip(models, X_test)]

    return y_pred, y_test
