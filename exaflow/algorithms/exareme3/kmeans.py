from typing import List

from pydantic import BaseModel

from exaflow.algorithms.exareme3.library.stats.stats import kmeans
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf

ALGORITHM_NAME = "kmeans"


class KMeansResult(BaseModel):
    title: str
    n_obs: int
    centers: List[List[float]]


class KMeansAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self):
        n_clusters = int(self.get_parameter("k"))

        tol = float(self.get_parameter("tol", 1e-4))
        maxiter = int(self.get_parameter("maxiter", 100))

        results = self.run_local_udf(
            func=local_step,
            kw_args={
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
def local_step(agg_client, data, n_clusters, tol, maxiter):
    result = kmeans(
        agg_client=agg_client,
        x=data,
        n_clusters=int(n_clusters),
        tol=float(tol),
        maxiter=int(maxiter),
    )
    return result
