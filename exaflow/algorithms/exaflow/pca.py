from typing import List
from typing import Sequence

import numpy as np
import pyarrow as pa
from pydantic import BaseModel

from exaflow.algorithms.exaflow.algorithm import Algorithm
from exaflow.algorithms.exaflow.cursor import Cursor
from exaflow.algorithms.exaflow.exaflow_registry import exaflow_udf
from exaflow.algorithms.exaflow.library.stats.stats import pca

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
def local_step(cursor: Cursor, inputdata, agg_client):
    """
    Execute PCA in streaming mode by passing batches of the requested y-columns
    to the underlying stats helper.
    """

    arrow_factory = cursor.arrow_streaming_factory()
    column_order = list(inputdata.y or [])

    def numpy_batch_factory():
        for table in arrow_factory():
            matrix = _table_to_numpy(table, column_order)
            if matrix is not None:
                yield matrix

    numpy_batch_factory.n_features = len(column_order)
    return pca(agg_client, numpy_batch_factory)


def _table_to_numpy(table: pa.Table, column_order: Sequence[str]) -> np.ndarray | None:
    """
    Convert an Arrow table batch to a NumPy 2D array without detouring through pandas.
    """

    selected = table.select(column_order) if column_order else table
    if selected.num_rows == 0 or selected.num_columns == 0:
        return None
    df = selected.to_pandas(split_blocks=True, self_destruct=True)
    if df.empty:
        return None
    return df.to_numpy(dtype=float, copy=False)
