from typing import List
from typing import Optional
from typing import TypeVar
from typing import Union

import numpy
from pydantic import BaseModel

from mipengine.algorithms.algorithm import Algorithm
from mipengine.algorithms.algorithm import AlgorithmDataLoader
from mipengine.algorithms.helpers import get_transfer_data
from mipengine.udfgen import MIN_ROW_COUNT
from mipengine.udfgen import literal
from mipengine.udfgen import merge_transfer
from mipengine.udfgen import relation
from mipengine.udfgen import secure_transfer
from mipengine.udfgen import transfer
from mipengine.udfgen import udf

S = TypeVar("S")

ALGORITHM_NAME = "random_forest"


class RandomForestDataLoader(AlgorithmDataLoader, algname=ALGORITHM_NAME):
    def get_variable_groups(self):
        return [self._variables.y + self._variables.x]


class RandomForestResult(BaseModel):
    title: str  # y


class RandomForestAlgorithm(Algorithm, algname=ALGORITHM_NAME):
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

        [data] = data

        metadata = dict(metadata)

        vars = [var for var in xvars + yvars]

        nominal_vars = [var for var in vars if metadata[var]["is_categorical"]]

        enumerations_dict = {var: metadata[var]["enumerations"] for var in nominal_vars}

        locals_result = local_run(
            func=fit_trees_local,
            positional_args=[data, enumerations_dict, yvar, xvars, num_trees],
            share_to_global=[True],
        )

        global_result = global_run(
            func=merge_trees,
            positional_args=[locals_result],
            share_to_locals=[True],
        )

        ret_val = RandomForestResult(title="random_forest")
        return ret_val


@udf(
    data=relation(S),
    enumerations_dict=literal(),
    yvar=literal(),
    xvars=literal(),
    num_trees=literal(),
    return_type=[transfer()],
)
def fit_trees_local(data, enumerations_dict, yvar, xvars, num_trees):
    import codecs
    import pickle

    from sklearn import preprocessing
    from sklearn.ensemble import RandomForestClassifier

    def serialize_rf(model):

        rf_serialized = pickle.dumps(model)
        curr_str = codecs.encode(rf_serialized, "base64").decode()

        return curr_str

    def deserialize_rf(curr_str):

        bytes_again = codecs.decode(curr_str.encode(), "base64")
        unpickled = pickle.loads(bytes_again)

        return unpickled

    X = data[xvars]
    y = data[yvar]
    le = preprocessing.LabelEncoder()
    le.fit(y)
    ydata_enc = le.transform(y)

    model = RandomForestClassifier(n_estimators=num_trees)
    model.fit(X, ydata_enc)

    transfer_ = {
        "forest": serialize_rf(model),
    }

    return transfer_


@udf(local_transfers=merge_transfer(), return_type=transfer())
def merge_trees(local_transfers):

    import codecs
    import pickle

    def serialize_rf(model):

        rf_serialized = pickle.dumps(model)
        curr_str = codecs.encode(rf_serialized, "base64").decode()

        return curr_str

    def deserialize_rf(curr_str):

        bytes_again = codecs.decode(curr_str.encode(), "base64")
        unpickled = pickle.loads(bytes_again)

        return unpickled

    forest_list = []
    for transfer in local_transfers:
        curr_model = transfer["forest"]
        curr_model2 = deserialize_rf(curr_model)
        forest_list.append(curr_model2)

    from copy import deepcopy

    final_model_rf = deepcopy(forest_list[0])
    list1 = []
    for curr_model2 in forest_list:
        estimators = curr_model2.estimators_
        for curr_estimator in estimators:
            list1.append(curr_estimator)
    final_model_rf.estimators_ = list1

    transfer_ = {"final_model": serialize_rf(final_model_rf)}

    return transfer_
