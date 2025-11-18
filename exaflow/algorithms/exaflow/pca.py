from typing import List

from pydantic import BaseModel

from exaflow.algorithms.exaflow.algorithm import Algorithm
from exaflow.algorithms.exaflow.exaflow_registry import exaflow_udf

ALGORITHM_NAME = "pca"


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
    from exaflow.worker.exaflow.duckdb import pca as duckdb_pca

    return duckdb_pca.run_pca(inputdata, agg_client)
