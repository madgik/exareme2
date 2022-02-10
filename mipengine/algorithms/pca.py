import json
from typing import TypeVar, List

import numpy
from pydantic import BaseModel

from mipengine.udfgen.udfgenerator import (
    udf,
    transfer,
    state,
    relation,
    merge_transfer,
)


class PCAResult(BaseModel):
    title: str
    n_obs: int
    eigenvalues: List[float]
    eigenvectors: List[List[float]]


def run(algo_interface):
    local_run = algo_interface.run_udf_on_local_nodes
    global_run = algo_interface.run_udf_on_global_node

    X_relation = algo_interface.initial_view_tables["y"]

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
    result = json.loads(result.get_table_data()[1][0])
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


@udf(x=relation(schema=S), return_type=[transfer()])
def local1(x):
    n_obs = len(x)
    sx = x.sum(axis=0)
    sxx = (x ** 2).sum(axis=0)

    transfer_ = {}
    transfer_["n_obs"] = n_obs
    transfer_["sx"] = sx.tolist()
    transfer_["sxx"] = sxx.tolist()
    return transfer_


@udf(local_transfers=merge_transfer(), return_type=[state(), transfer()])
def global1(local_transfers):
    n_obs = sum(t["n_obs"] for t in local_transfers)
    sx = sum(numpy.array(t["sx"]) for t in local_transfers)
    sxx = sum(numpy.array(t["sxx"]) for t in local_transfers)

    means = sx / n_obs
    sigmas = ((sxx - n_obs * means ** 2) / (n_obs - 1)) ** 0.5

    state_ = dict(n_obs=n_obs)
    transfer_ = dict(means=means.tolist(), sigmas=sigmas.tolist())
    return state_, transfer_


@udf(x=relation(schema=S), global_transfer=transfer(), return_type=[transfer()])
def local2(x, global_transfer):
    means = numpy.array(global_transfer["means"])
    sigmas = numpy.array(global_transfer["sigmas"])

    x -= means
    x /= sigmas
    gramian = x.T @ x

    transfer_ = dict(gramian=gramian.values.tolist())
    return transfer_


@udf(local_transfers=merge_transfer(), prev_state=state(), return_type=[transfer()])
def global2(local_transfers, prev_state):
    gramian = sum(numpy.array(t["gramian"]) for t in local_transfers)
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
