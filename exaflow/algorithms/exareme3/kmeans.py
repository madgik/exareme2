from typing import List

from pydantic import BaseModel

from exaflow.algorithms.exareme3.algorithm import Algorithm
from exaflow.algorithms.exareme3.exareme3_registry import exareme3_udf
from exaflow.algorithms.exareme3.library.stats.stats import kmeans

ALGORITHM_NAME = "kmeans"


class KMeansResult(BaseModel):
    title: str
    n_obs: int
    centers: List[List[float]]


class KMeansAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata):
        n_clusters = int(self.parameters["k"])
        # Optional parameters with defaults
        tol = float(self.parameters.get("tol", 1e-4))
        maxiter = int(self.parameters.get("maxiter", 100))

        results = self.engine.run_algorithm_udf(
            func=local_step,
            positional_args={
                "inputdata": self.inputdata.json(),
                "n_clusters": n_clusters,
                "tol": tol,
                "maxiter": maxiter,
            },
        )

        result = results[0]
        n_obs = int(result["n_obs"])
        centers = result["centers"]

        return KMeansResult(
            title="K-Means Centers",
            n_obs=n_obs,
            centers=centers,
        )


@exareme3_udf(with_aggregation_server=True)
def local_step(data, inputdata, agg_client, n_clusters, tol, maxiter):
    """
    Local exaflow UDF wrapper:

    - Fetch local data (same way as PCA).
    - Select the columns in inputdata.y as the feature matrix.
    - Delegate the actual distributed K-means to `stats.kmeans`, which uses
      `agg_client` to aggregate across workers.
    """

    # `inputdata.y` is a list of variable names; we use them as features
    # just like PCA does: data[inputdata.y] is a 2D array-like.
    X = data[inputdata.y]

    # `kmeans` is expected to implement the distributed logic using agg_client
    # and return a dict, e.g.:
    #   {"n_obs": int, "centers": List[List[float]]}
    result = kmeans(
        agg_client=agg_client,
        x=X,
        n_clusters=int(n_clusters),
        tol=float(tol),
        maxiter=int(maxiter),
    )
    return result
