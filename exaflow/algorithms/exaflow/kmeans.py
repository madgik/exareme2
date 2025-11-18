from typing import List

from pydantic import BaseModel

from exaflow.algorithms.exaflow.algorithm import Algorithm
from exaflow.algorithms.exaflow.exaflow_registry import exaflow_udf
from exaflow.algorithms.exaflow.library.stats.stats import kmeans
from exaflow.worker_communication import BadUserInput

ALGORITHM_NAME = "kmeans"


class KMeansResult(BaseModel):
    title: str
    n_obs: int
    centers: List[List[float]]


class KMeansAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata):
        # We expect clustering variables in y (same as the exaflow version)
        if not self.inputdata.y:
            raise BadUserInput("K-means requires at least one variable in 'y'.")

        use_duckdb = True

        # Required parameters
        try:
            n_clusters = int(self.parameters["k"])
        except (KeyError, ValueError, TypeError):
            raise BadUserInput(
                "Parameter 'k' (number of clusters) must be provided as an integer."
            )

        # Optional parameters with defaults
        tol = float(self.parameters.get("tol", 1e-4))
        maxiter = int(self.parameters.get("maxiter", 100))

        # Run the exaflow UDF
        results = self.engine.run_algorithm_udf(
            func=local_step,
            positional_args={
                "inputdata": self.inputdata.json(),
                "n_clusters": n_clusters,
                "tol": tol,
                "maxiter": maxiter,
                "use_duckdb": use_duckdb,
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


@exaflow_udf(with_aggregation_server=True)
def local_step(inputdata, csv_paths, agg_client, n_clusters, tol, maxiter, use_duckdb):
    """
    Local exaflow UDF wrapper:

    - Fetch local data (same way as PCA).
    - Select the columns in inputdata.y as the feature matrix.
    - Delegate the actual distributed K-means to `stats.kmeans`, which uses
      `agg_client` to aggregate across workers.
    """
    from exaflow.algorithms.exaflow.data_loading import load_algorithm_dataframe

    data = load_algorithm_dataframe(inputdata, csv_paths, dropna=True)

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
