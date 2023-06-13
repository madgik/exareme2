from typing import List
from typing import Optional
from typing import TypeVar
from typing import Union

import numpy
from pydantic import BaseModel

from mipengine.algorithms.algorithm import Algorithm
from mipengine.algorithms.algorithm import AlgorithmDataLoader
from mipengine.algorithms.crossvalidation import KFold
from mipengine.algorithms.helpers import get_transfer_data
from mipengine.algorithms.random_forest import *
from mipengine.udfgen import MIN_ROW_COUNT
from mipengine.udfgen import literal
from mipengine.udfgen import merge_transfer
from mipengine.udfgen import relation
from mipengine.udfgen import secure_transfer
from mipengine.udfgen import transfer
from mipengine.udfgen import udf

S = TypeVar("S")

ALGORITHM_NAME = "random_forest_cv"


class RandomForestCVDataLoader(AlgorithmDataLoader, algname=ALGORITHM_NAME):
    def get_variable_groups(self):
        return [self._variables.x, self._variables.y]


class RandomForestCVResult(BaseModel):
    title: str  # y
    accuracy_list: List[float]


class RandomForestCVAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, data, metadata):

        local_run = self.engine.run_udf_on_local_nodes
        global_run = self.engine.run_udf_on_global_node

        xvars = self.variables.x
        yvars = self.variables.y

        yvar = yvars[0]

        default_estimators = 64
        num_trees = self.algorithm_parameters.get("num_estimators", default_estimators)
        if num_trees is None:
            num_trees = default_estimators

        default_splits = 5
        n_splits = self.algorithm_parameters.get("n_splits", default_splits)
        if n_splits is None:
            n_splits = default_splits

        kf = KFold(self.engine, n_splits=n_splits)

        X, y = data

        metadata = dict(metadata)

        vars = [var for var in xvars + yvars]

        nominal_vars = [var for var in vars if metadata[var]["is_categorical"]]

        enumerations_dict = {var: metadata[var]["enumerations"] for var in nominal_vars}

        X_train, X_test, y_train, y_test = kf.split(X, y)

        curr_accuracy_list = []

        for curr_x_train, curr_y_train, curr_x_test, curr_y_test in zip(
            X_train, y_train, X_test, y_test
        ):
            # model.fit(X=X, y=y)

            locals_result = local_run(
                func=fit_trees_local,
                positional_args=[
                    curr_x_train,
                    curr_y_train,
                    enumerations_dict,
                    yvar,
                    xvars,
                    num_trees,
                ],
                share_to_global=[True],
            )

            global_result = global_run(
                func=merge_trees,
                positional_args=[locals_result],
                share_to_locals=[True],
            )

            locals_result2 = local_run(
                func=predict_trees_local,
                positional_args=[
                    curr_x_test,
                    curr_y_test,
                    enumerations_dict,
                    yvar,
                    xvars,
                    global_result,
                ],
                share_to_global=[True],
            )

            global_accuracy = global_run(
                func=compute_accuracy,
                positional_args=[locals_result2],
                share_to_locals=[False],
            )

            curr_accuracy = get_transfer_data(global_accuracy)["total_accuracy"]
            curr_accuracy_list.append(curr_accuracy)

        ret_val = RandomForestCVResult(
            title="random_forest", accuracy_list=curr_accuracy_list
        )
        return ret_val
