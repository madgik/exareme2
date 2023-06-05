from typing import List
from typing import Optional
from typing import TypeVar
from typing import Union

import numpy
from pydantic import BaseModel

from mipengine.algorithm_specification import AlgorithmSpecification
from mipengine.algorithm_specification import InputDataSpecification
from mipengine.algorithm_specification import InputDataSpecifications
from mipengine.algorithm_specification import InputDataStatType
from mipengine.algorithm_specification import InputDataType
from mipengine.algorithm_specification import ParameterSpecification
from mipengine.algorithm_specification import ParameterType
from mipengine.algorithms.algorithm import Algorithm
from mipengine.algorithms.helpers import get_transfer_data
from mipengine.udfgen import MIN_ROW_COUNT
from mipengine.udfgen import literal
from mipengine.udfgen import merge_transfer
from mipengine.udfgen import relation
from mipengine.udfgen import secure_transfer
from mipengine.udfgen import transfer
from mipengine.udfgen import udf

S = TypeVar("S")


class MultipleHistogramsAlgorithm(Algorithm, algname="random_forest"):
    @classmethod
    def get_specification(cls):
        return AlgorithmSpecification(
            name=cls.algname,
            desc="Random Forest",
            label="Random Forest",
            enabled=True,
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="y",
                    desc="class_values",
                    types=[InputDataType.INT, InputDataType.TEXT],
                    stattypes=[InputDataStatType.NOMINAL],
                    notblank=True,
                    multiple=False,
                ),
                x=InputDataSpecification(
                    label="x",
                    desc="Nominal variable for grouping bins.",
                    types=[InputDataType.REAL, InputDataType.INT, InputDataType.TEXT],
                    stattypes=[InputDataStatType.NUMERICAL, InputDataStatType.NOMINAL],
                    notblank=False,
                    multiple=True,
                ),
            ),
            parameters={
                "n_estimators": ParameterSpecification(
                    label="Number of Trees",
                    desc="Number of Trees",
                    types=[ParameterType.INT],
                    notblank=False,
                    multiple=False,
                    default=64,
                    min=1,
                    max=100,
                ),
            },
        )

    def get_variable_groups(self):
        return [self.variables.y + self.variables.x]

    def run(self, engine):
        local_run = engine.run_udf_on_local_nodes
        global_run = engine.run_udf_on_global_node

        xvars = self.variables.x
        yvars = self.variables.y

        yvar = yvars[0]

        default_estimators = 64
        num_trees = self.algorithm_parameters.get("num_estimators", default_estimators)
        if num_trees is None:
            num_trees = default_bins

        [data] = engine.data_model_views

        metadata = dict(self.metadata)

        vars = [var for var in xvars + yvars]

        nominal_vars = [var for var in vars if metadata[var]["is_categorical"]]

        enumerations_dict = {var: metadata[var]["enumerations"] for var in nominal_vars}

        locals_result = local_run(
            func=fit_trees_local,
            positional_args=[data, enumerations_dict, yvar, xvars],
            share_to_global=[True],
        )

        global_result = global_run(
            func=merge_trees,
            positional_args=[locals_result],
            share_to_locals=[True],
        )

    @udf(
        data=relation(S),
        enumerations_dict=literal(),
        yvar=literal(),
        xvars=literal(),
        num_trees=literal(),
        return_type=[transfer()],
    )
    def fit_trees_local(data, enumerations_dict, yvar, xvars, num_trees):
        from sklearn.ensemble import RandomForestClassifier

        xdata = data[xvars]
        ydata = data[yvar]

        model = RandomForestClassifier(n_estimators=num_trees)
        model.fit(xdata, ydata)

        transfer_ = {
            "forest": serialize_rf(model),
        }

        return transfer_

    @udf(local_transfers=merge_transfer(), return_type=transfer())
    def merge_trees(local_transfers):
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

    def serialize_rf(model):
        import pickle

        curr_bytes = pickle.dumps(model)
        curr_str = curr_bytes.decode("utf-8")
        return curr_str

        # return bytes_list

    def deserialize_rf(model_str):
        import pickle

        curr_byte = model_str.encode("utf-8")
        curr_model = pickle.loads(curr_byte)

        return curr_model
