import warnings
from typing import List

from pydantic import BaseModel

from exareme2.algorithms.exaflow.algorithm import Algorithm
from exareme2.algorithms.exaflow.exaflow_registry import exaflow_udf

ALGORITHM_NAME = "pca_exaflow_aggregator"


class PCAResult(BaseModel):
    title: str
    n_obs: int
    eigenvalues: List[float]
    eigenvectors: List[List[float]]


class PCAAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata):
        results = self.engine.run_algorithm_udf(
            func=local_step,
            positional_args={"inputdata": self.inputdata.json()},
        )
        result = results[0]
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


def local1(agg_client, x):
    import numpy

    n_obs = len(x)
    sx = numpy.einsum("ij->j", x)
    sxx = numpy.einsum("ij,ij->j", x, x)

    total_n_obs = agg_client.sum([float(n_obs)])[0]
    total_sx = numpy.array(agg_client.sum(sx.tolist()), dtype=float)
    total_sxx = numpy.array(agg_client.sum(sxx.tolist()), dtype=float)

    if total_n_obs <= 1:
        raise ValueError("PCA requires at least two observations across all workers.")

    means = total_sx / total_n_obs
    variances = (total_sxx - total_n_obs * means**2) / (total_n_obs - 1)
    variances = numpy.maximum(variances, 0.0)
    sigmas = numpy.sqrt(variances)
    zero_sigma = sigmas == 0
    if numpy.any(zero_sigma):
        sigmas = sigmas.copy()
        sigmas[zero_sigma] = 1.0
    x = x.values
    out = numpy.empty(x.shape)

    numpy.subtract(x, means, out=out)
    numpy.divide(out, sigmas, out=out)
    gramian = numpy.einsum("ji,jk->ik", out, out)
    total_gramian = numpy.array(agg_client.sum(gramian.tolist()), dtype=float)
    covariance = total_gramian / (total_n_obs - 1)

    eigenvalues, eigenvectors = numpy.linalg.eig(covariance)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    eigenvectors = eigenvectors.T
    return dict(
        n_obs=int(total_n_obs),
        eigenvalues=eigenvalues.real.tolist(),
        eigenvectors=eigenvectors.real.tolist(),
    )


@exaflow_udf(with_aggregation_server=True)
def local_step(inputdata, csv_paths, agg_client):
    from exareme2.algorithms.utils.inputdata_utils import fetch_data

    data = fetch_data(inputdata, csv_paths)

    return local1(agg_client, data[inputdata.y])
