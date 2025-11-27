from typing import List

import numpy as np
from pydantic import BaseModel
from sklearn.svm import SVC

from exaflow.algorithms.exaflow.algorithm import Algorithm
from exaflow.algorithms.exaflow.exaflow_registry import exaflow_udf
from exaflow.worker_communication import BadUserInput

ALGORITHM_NAME = "svm_scikit"


class SVMResult(BaseModel):
    title: str
    n_obs: int
    coeff: List[float]
    support_vectors: List[float]


class SVMAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata: dict):
        if not self.inputdata.y:
            raise BadUserInput("SVM requires one dependent variable (y).")
        if not self.inputdata.x:
            raise BadUserInput("SVM requires at least one covariate (x).")

        y_var = self.inputdata.y[0]
        x_vars = self.inputdata.x

        gamma = self.parameters.get("gamma")
        C = self.parameters.get("C")
        if gamma is None or C is None:
            raise BadUserInput("Parameters 'gamma' and 'C' are required for SVM.")

        # Validate that y has at least two levels using metadata enumerations
        y_enums = metadata.get(y_var, {}).get("enumerations")
        if not y_enums:
            raise BadUserInput(
                f"Covariate '{y_var}' must be categorical with enumerations."
            )
        y_levels = list(y_enums.keys())
        if len(y_levels) < 2:
            raise BadUserInput(
                f"The variable {y_var} has less than 2 levels and SVM cannot be "
                "performed. Please choose another variable."
            )

        udf_results = self.engine.run_algorithm_udf(
            func=svm_scikit_local_step,
            positional_args={
                "inputdata": self.inputdata.json(),
                "y_var": y_var,
                "x_vars": x_vars,
                "y_levels": y_levels,
                "gamma": float(gamma),
                "C": float(C),
            },
        )

        model_stats = udf_results[0]
        return SVMResult(
            title="SVM Result",
            n_obs=int(model_stats["n_obs"]),
            coeff=model_stats["coeff"],
            support_vectors=model_stats["support_vectors"],
        )


@exaflow_udf(with_aggregation_server=True)
def svm_scikit_local_step(
    data, inputdata, agg_client, y_var, x_vars, y_levels, gamma, C
):
    """
    Train a linear SVM locally, then securely average the model parameters
    (coefficients and a per-feature summary of support vectors) across workers.
    """
    # Keep only required columns and drop rows with missing values
    cols = list(dict.fromkeys(list(x_vars) + [y_var]))
    data = data[cols].dropna()

    n_features = len(x_vars)
    if n_features == 0:
        raise BadUserInput("SVM requires at least one covariate (x).")

    X = data[x_vars].to_numpy(dtype=float, copy=False)
    y = data[y_var].to_numpy(copy=False)

    n_obs_local = float(len(y))
    unique_y = np.unique(y)
    if unique_y.size < 2:
        raise BadUserInput("Cannot perform SVM. Covariable has only one level.")

    model = SVC(kernel="linear", gamma=gamma, C=C)
    model.fit(X, y)

    coeff_arr = np.asarray(model.coef_, dtype=float)
    # Fix shape across workers: average across class rows to a single vector
    coeff_local = coeff_arr.mean(axis=0)

    # Summarize support vectors as mean per feature to keep a fixed shape.
    support_summary_local = np.asarray(model.support_vectors_, dtype=float).mean(axis=0)

    coeff_sum_arr = agg_client.sum(coeff_local)
    support_sum_arr = agg_client.sum(support_summary_local)
    n_obs_arr = agg_client.sum(np.array([n_obs_local], dtype=float))
    workers_arr = agg_client.sum(np.array([1.0], dtype=float))

    num_workers = float(np.asarray(workers_arr, dtype=float).reshape(-1)[0] or 1.0)
    coeff_mean = np.asarray(coeff_sum_arr, dtype=float) / num_workers
    support_mean = np.asarray(support_sum_arr, dtype=float) / num_workers
    total_n_obs = int(np.asarray(n_obs_arr, dtype=float).reshape(-1)[0])

    return {
        "n_obs": total_n_obs,
        "coeff": coeff_mean.reshape(-1).tolist(),
        "support_vectors": support_mean.reshape(-1).tolist(),
    }
