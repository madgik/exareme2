from typing import Dict
from typing import List

import numpy as np
from pydantic import BaseModel

from exaflow.algorithms.exaflow.algorithm import Algorithm
from exaflow.algorithms.exaflow.exaflow_registry import exaflow_udf
from exaflow.algorithms.exaflow.library.stats.stats import pca as core_pca
from exaflow.worker_communication import BadUserInput

ALGORITHM_NAME = "pca_with_transformation"


class PCAResult(BaseModel):
    title: str
    n_obs: int
    eigenvalues: List[float]
    eigenvectors: List[List[float]]


class PCAWithTransformationAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata: dict):
        """
        PCA with data transformations, exaflow-style.

        It mirrors the old exaflow PCA-with-transformation behaviour:

        1. Optionally applies local log/exp to selected columns.
        2. Optionally applies global center/standardize (via agg_client) to
           selected columns.
        3. Always calls the base `pca` helper, which standardizes all columns
           and computes covariance eigen-decomposition.

        Error messages are preserved so the validation tests match.
        """
        data_transformation: Dict = self.parameters.get("data_transformation", {})
        use_duckdb = True

        try:
            results = self.engine.run_algorithm_udf(
                func=pca_with_transformation_local_step,
                positional_args={
                    "inputdata": self.inputdata.json(),
                    "data_transformation": data_transformation,
                    "use_duckdb": use_duckdb,
                },
            )
        except Exception as ex:
            msg = str(ex)
            if (
                "Log transformation cannot be applied to non-positive values in column"
                in msg
                or "Unknown transformation" in msg
                or "Standardization cannot be applied to column" in msg
            ):
                # Same user-facing behaviour as old exaflow implementation
                raise BadUserInput(msg)
            raise

        result = results[0]
        return PCAResult(
            title="Eigenvalues and Eigenvectors",
            n_obs=result["n_obs"],
            eigenvalues=result["eigenvalues"],
            eigenvectors=result["eigenvectors"],
        )


@exaflow_udf(with_aggregation_server=True)
def pca_with_transformation_local_step(
    inputdata,
    csv_paths,
    agg_client,
    data_transformation,
    use_duckdb,
):
    """
    UDF that:
      - fetches data
      - applies log/exp locally
      - if requested, computes global means/sigmas for center/standardize
      - applies those transforms
      - then delegates to the standard `pca` helper

    data_transformation is expected to look like:

        {
            "log": ["col1", "col2"],
            "exp": ["col3"],
            "center": ["col4"],
            "standardize": ["col5", "col6"],
        }
    """
    import pandas as pd

    from exaflow.algorithms.exaflow.data_loading import load_algorithm_dataframe

    if data_transformation is None:
        data_transformation = {}

    # Validate keys (mirror old behaviour)
    allowed_keys = {"log", "exp", "center", "standardize"}
    for key in data_transformation.keys():
        if key not in allowed_keys:
            raise ValueError(f"Unknown transformation: {key}")

    data = load_algorithm_dataframe(inputdata, csv_paths, dropna=True)

    # Use the same y variables as the base PCA implementation
    y_vars = inputdata.y
    X = data[y_vars].copy()

    # ---------------------------
    # 1. Local log / exp
    # ---------------------------
    log_cols = data_transformation.get("log", []) or []
    exp_cols = data_transformation.get("exp", []) or []

    for col in log_cols:
        if col not in X.columns:
            continue
        if (X[col] <= 0).any():
            # Exact wording is important for tests
            raise ValueError(
                f"Log transformation cannot be applied to non-positive values in column '{col}'."
            )
        X[col] = np.log(X[col])

    for col in exp_cols:
        if col not in X.columns:
            continue
        X[col] = np.exp(X[col])

    # ---------------------------
    # 2. Global center / standardize (optional)
    # ---------------------------
    center_cols = set(data_transformation.get("center", []) or [])
    standardize_cols = set(data_transformation.get("standardize", []) or [])

    if center_cols or standardize_cols:
        # Compute global stats across all workers on the *log/exp-transformed* X
        if isinstance(X, pd.DataFrame):
            X_values = X.to_numpy(dtype=float)
        else:
            X_values = np.asarray(X, dtype=float)

        n_obs_local = float(X_values.shape[0])
        sx_local = (
            np.einsum("ij->j", X_values)
            if n_obs_local > 0
            else np.zeros(X_values.shape[1], dtype=float)
        )
        sxx_local = (
            np.einsum("ij,ij->j", X_values, X_values)
            if n_obs_local > 0
            else np.zeros(X_values.shape[1], dtype=float)
        )

        total_n_obs = float(agg_client.sum([n_obs_local])[0])
        if total_n_obs <= 1:
            # Degenerate case, keep things as-is (no meaningful stats)
            means = np.zeros_like(sx_local)
            sigmas = np.ones_like(sx_local)
        else:
            total_sx = np.asarray(agg_client.sum(sx_local.tolist()), dtype=float)
            total_sxx = np.asarray(agg_client.sum(sxx_local.tolist()), dtype=float)

            means = total_sx / total_n_obs
            variances = (total_sxx - total_n_obs * means**2) / (total_n_obs - 1)
            variances = np.maximum(variances, 0.0)
            sigmas = np.sqrt(variances)

        # Map column name -> index
        col_to_idx = {name: idx for idx, name in enumerate(X.columns)}

        # Check explicit standardize columns for zero sigma
        for col in standardize_cols:
            if col not in col_to_idx:
                continue
            j = col_to_idx[col]
            if sigmas[j] == 0:
                raise ValueError(
                    f"Standardization cannot be applied to column '{col}' because its standard deviation is zero."
                )

        # Apply center / standardize using global stats
        if X_values.size > 0:
            for col_name, j in col_to_idx.items():
                if col_name in standardize_cols:
                    X_values[:, j] = (X_values[:, j] - means[j]) / sigmas[j]
                elif col_name in center_cols:
                    X_values[:, j] = X_values[:, j] - means[j]

            # Push back into DataFrame (preserve column order)
            X.iloc[:, :] = X_values

    # ---------------------------
    # 3. Delegate to the base PCA
    # ---------------------------
    # The base PCA helper will:
    #   - compute global means/sigmas again
    #   - standardize all columns
    #   - compute covariance and eigen-decomposition
    # This matches the old exaflow PCA-with-transformation behaviour.
    return core_pca(agg_client, X)
