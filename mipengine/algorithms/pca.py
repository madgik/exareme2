from typing import List
from typing import TypeVar

import numpy
from pydantic import BaseModel

from mipengine.algorithms.base_classes.algorithm import Algorithm
from mipengine.algorithms.helpers import get_transfer_data
from mipengine.algorithms.specifications.algorithm_specification import AlgorithmName
from mipengine.algorithms.specifications.algorithm_specification import (
    AlgorithmSpecification,
)
from mipengine.algorithms.specifications.inputdata_specification import (
    InputDataSpecification,
)
from mipengine.algorithms.specifications.inputdata_specification import (
    InputDataSpecifications,
)
from mipengine.algorithms.specifications.inputdata_specification import (
    InputDataStatType,
)
from mipengine.algorithms.specifications.inputdata_specification import InputDataType
from mipengine.udfgen import relation
from mipengine.udfgen import secure_transfer
from mipengine.udfgen import state
from mipengine.udfgen import transfer
from mipengine.udfgen import udf


class PCAResult(BaseModel):
    title: str
    n_obs: int
    eigenvalues: List[float]
    eigenvectors: List[List[float]]


class PCAAlgorithm(Algorithm, stepname=AlgorithmName.PCA.value):
    @classmethod
    def get_specification(cls):
        return AlgorithmSpecification(
            name=cls.stepname,
            desc="PCA",
            label="PCA",
            enabled=True,
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="Variables",
                    desc="Variables",
                    types=[InputDataType.REAL],
                    stattypes=[InputDataStatType.NUMERICAL],
                    notblank=True,
                    multiple=True,
                ),
            ),
        )

    def get_variable_groups(self):
        return [self.variables.y]

    def run(self, engine):
        local_run = engine.run_udf_on_local_nodes
        global_run = engine.run_udf_on_global_node

        [X_relation] = engine.data_model_views

        local_transfers = local_run(
            func=local1,
            keyword_args={"x": X_relation},
            share_to_global=[True],
        )
        global_state, global_transfer = global_run(
            func=global1,
            keyword_args=dict(local_transfers=local_transfers),
            share_to_locals=[False, True],
        )
        local_transfers = local_run(
            func=local2,
            keyword_args=dict(x=X_relation, global_transfer=global_transfer),
            share_to_global=[True],
        )
        result = global_run(
            func=global2,
            keyword_args=dict(local_transfers=local_transfers, prev_state=global_state),
        )
        result = get_transfer_data(result)
        n_obs = result["n_obs"]
        eigenvalues = result["eigenvalues"]
        eigenvectors = result["eigenvectors"]

        result = PCAResult(
            title="Eigenvalues and Eigenvectors",
            n_obs=n_obs,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
        )
        return result


S = TypeVar("S")


@udf(x=relation(schema=S), return_type=[secure_transfer(sum_op=True)])
def local1(x):
    n_obs = len(x)
    sx = numpy.einsum("ij->j", x)
    sxx = numpy.einsum("ij,ij->j", x, x)

    transfer_ = {}
    transfer_["n_obs"] = {"data": n_obs, "operation": "sum", "type": "int"}
    transfer_["sx"] = {"data": sx.tolist(), "operation": "sum", "type": "float"}
    transfer_["sxx"] = {"data": sxx.tolist(), "operation": "sum", "type": "float"}
    return transfer_


@udf(local_transfers=secure_transfer(sum_op=True), return_type=[state(), transfer()])
def global1(local_transfers):
    n_obs = local_transfers["n_obs"]
    sx = numpy.array(local_transfers["sx"])
    sxx = numpy.array(local_transfers["sxx"])

    means = sx / n_obs
    sigmas = ((sxx - n_obs * means**2) / (n_obs - 1)) ** 0.5

    state_ = dict(n_obs=n_obs)
    transfer_ = dict(means=means.tolist(), sigmas=sigmas.tolist())
    return state_, transfer_


@udf(
    x=relation(schema=S),
    global_transfer=transfer(),
    return_type=[secure_transfer(sum_op=True)],
)
def local2(x, global_transfer):
    means = numpy.array(global_transfer["means"])
    sigmas = numpy.array(global_transfer["sigmas"])

    x = x.values
    out = numpy.empty(x.shape)

    numpy.subtract(x, means, out=out)
    numpy.divide(out, sigmas, out=out)
    gramian = numpy.einsum("ji,jk->ik", out, out)

    transfer_ = {
        "gramian": {
            "data": gramian.tolist(),
            "operation": "sum",
            "type": "float",
        }
    }
    return transfer_


@udf(
    local_transfers=secure_transfer(sum_op=True),
    prev_state=state(),
    return_type=[transfer()],
)
def global2(local_transfers, prev_state):
    gramian = numpy.array(local_transfers["gramian"])
    n_obs = prev_state["n_obs"]
    covariance = gramian / (n_obs - 1)

    eigenvalues, eigenvectors = numpy.linalg.eig(covariance)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    eigenvectors = eigenvectors.T

    transfer_ = dict(
        n_obs=n_obs,
        eigenvalues=eigenvalues.tolist(),
        eigenvectors=eigenvectors.tolist(),
    )
    return transfer_
