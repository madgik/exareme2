from typing import List

import numpy
from pydantic import BaseModel

from exareme2.aggregator.constants import AggregationType
from exareme2.algorithms.exaflow.aggregator_client import AggregationClient
from exareme2.algorithms.exaflow.algorithm import Algorithm
from exareme2.algorithms.exaflow.exaflow_registry import exaflow_udf

ALGORITHM_NAME = "pca"


class PCAResult(BaseModel):
    title: str
    n_obs: int
    eigenvalues: List[float]
    eigenvectors: List[List[float]]


class PCAAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata):
        result = self.engine.run_algorithm_udf_with_aggregator(
            func="pca_run_algorithm",
            positional_args={"inputdata": self.inputdata.json()},
        )
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


@exaflow_udf
def compute_pca(agg_client, x):
    n_obs = len(x)
    n_obs = agg_client.aggregate(AggregationType.SUM, n_obs)[0]
    sx = agg_client.aggregate(AggregationType.SUM, numpy.einsum("ij->j", x))
    sxx = agg_client.aggregate(AggregationType.SUM, numpy.einsum("ij,ij->j", x, x))
    sx = numpy.array(sx)
    sxx = numpy.array(sxx)
    n_obs = int(n_obs)
    means = sx / n_obs
    sigmas = ((sxx - n_obs * means**2) / (n_obs - 1)) ** 0.5

    x = x.values
    out = numpy.empty(x.shape)

    numpy.subtract(x, means, out=out)
    numpy.divide(out, sigmas, out=out)

    gramian = numpy.einsum("ji,jk->ik", out, out)
    gramian = agg_client.aggregate(AggregationType.SUM, gramian.tolist())
    gramian = numpy.array(gramian)
    covariance = gramian / (int(n_obs) - 1)
    eigenvalues, eigenvectors = numpy.linalg.eig(covariance)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    eigenvectors = eigenvectors.T
    return dict(
        n_obs=n_obs,
        eigenvalues=eigenvalues.tolist(),
        eigenvectors=eigenvectors.tolist(),
    )


@exaflow_udf
def run_algorithm(request_id, inputdata, csv_paths):
    from exareme2.algorithms.utils.inputdata_utils import fetch_data

    data = fetch_data(inputdata, csv_paths)
    agg_client = AggregationClient(request_id=request_id)
    return compute_pca(agg_client, data[inputdata.y])
