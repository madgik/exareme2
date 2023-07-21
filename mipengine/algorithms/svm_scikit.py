from typing import List
from typing import TypeVar

from pydantic import BaseModel

from mipengine.algorithms.algorithm import Algorithm
from mipengine.algorithms.algorithm import AlgorithmDataLoader
from mipengine.algorithms.fedaverage import fed_average
from mipengine.algorithms.helpers import get_transfer_data
from mipengine.algorithms.helpers import sum_secure_transfers
from mipengine.exceptions import BadUserInput
from mipengine.udfgen import literal
from mipengine.udfgen import relation
from mipengine.udfgen import secure_transfer
from mipengine.udfgen import udf

ALGORITHM_NAME = "svm_scikit"


class SVMDataLoader(AlgorithmDataLoader, algname=ALGORITHM_NAME):
    def get_variable_groups(self):
        return [self._variables.x, self._variables.y]


class SVMResult(BaseModel):
    title: str
    n_obs: int
    coeff: List[float]
    support_vectors: List[float]


class SVMAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, data, metadata):
        X, y = data
        gamma = self.algorithm_parameters["gamma"]
        C = self.algorithm_parameters["C"]

        y_name = y.columns[0]
        y_enums = metadata[y_name]["enumerations"].keys()

        if len(y_enums) < 2:
            raise BadUserInput(
                f"The variable {y_name} has less than 2 levels and SVM cannot be "
                "performed. Please choose another variable."
            )

        models = SVMFedAverage(self.engine)
        models.fit(X, y, gamma, C)

        result = SVMResult(
            title="SVM Result",
            n_obs=models.nobs_train,
            coeff=models.coeff,
            support_vectors=models.support_vectors,
        )

        return result


S = TypeVar("S")


class SVMFedAverage:
    def __init__(self, engine):
        self.num_local_nodes = engine.num_local_nodes
        self.local_run = engine.run_udf_on_local_nodes
        self.global_run = engine.run_udf_on_global_node

    def fit(self, x, y, gamma, C):
        params_to_average, other_params = self.local_run(
            func=self._fit_local,
            keyword_args={"x": x, "y": y, "gamma": gamma, "C": C},
            share_to_global=[True, True],
        )
        averaged_params_table = self.global_run(
            func=fed_average,
            keyword_args=dict(
                params=params_to_average, num_local_nodes=self.num_local_nodes
            ),
        )
        other_params_table = self.global_run(
            func=sum_secure_transfers,
            keyword_args=dict(loctransf=other_params),
        )

        averaged_params = get_transfer_data(averaged_params_table)
        other_params = get_transfer_data(other_params_table)

        self.coeff = averaged_params["coeff"]
        self.support_vectors = averaged_params["support_vectors"]
        self.nobs_train = other_params["nobs_train"]

    @staticmethod
    @udf(
        x=relation(schema=S),
        y=relation(schema=S),
        gamma=literal(),
        C=literal(),
        return_type=[secure_transfer(sum_op=True), secure_transfer(sum_op=True)],
    )
    def _fit_local(x, y, gamma, C):
        import numpy as np
        from sklearn.svm import SVC

        n_obs = y.shape[0]
        y = y.to_numpy()
        X = x.to_numpy()

        y_unq = np.unique(y)
        if len(y_unq) < 2:
            raise ValueError("Cannot perform SVM. Covariable has only one level.")

        model = SVC(kernel="linear", gamma=gamma, C=C)
        model.fit(X, y)

        if len(model.coef_) < 2:
            coeff = [model.coef_.squeeze().tolist()]
        else:
            coeff = model.coef_.squeeze().tolist()

        params_to_average = {}
        params_to_average["coeff"] = {
            "data": coeff,
            "operation": "sum",
            "type": "float",
        }
        params_to_average["support_vectors"] = {
            "data": model.support_vectors_.squeeze().tolist(),
            "operation": "sum",
            "type": "float",
        }

        other_params = {
            "nobs_train": {
                "data": len(y),
                "operation": "sum",
                "type": "int",
            }
        }  # other quantities not meant to be averaged

        return params_to_average, other_params
