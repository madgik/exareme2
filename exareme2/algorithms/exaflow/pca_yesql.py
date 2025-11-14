import warnings
from typing import List

from pydantic import BaseModel

from exareme2.algorithms.exaflow.algorithm import Algorithm
from exareme2.algorithms.exaflow.exaflow_registry import exaflow_udf

ALGORITHM_NAME = "pca_yesql"


class PCAResult(BaseModel):
    title: str
    n_obs: int
    eigenvalues: List[float]
    eigenvectors: List[List[float]]


class PCAResult(BaseModel):
    title: str
    n_obs: int
    eigenvalues: List[float]
    eigenvectors: List[List[float]]


class PCAAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata):
        results = self.engine.run_algorithm_yesql(
            func="pca",
            positional_args={"inputdata": self.inputdata.json()},
        )
        ### For this to run the folder of YESQL should be present at /home/YeSQL to mirror the space that it exists on the docker.
        warnings.warn(f"{results=}")

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
