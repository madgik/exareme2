from typing import List

from pydantic import BaseModel

from exareme2.algorithms.exaflow.algorithm import Algorithm
from exareme2.algorithms.exaflow.exaflow_registry import exaflow_udf
from exareme2.algorithms.exaflow.library.stats.stats import pca

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


@exaflow_udf(with_aggregation_server=True)
def local_step(inputdata, csv_paths, agg_client):
    from exareme2.algorithms.utils.inputdata_utils import fetch_data

    data = fetch_data(inputdata, csv_paths)

    return pca(agg_client, data[inputdata.y])
