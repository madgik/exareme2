from typing import List
from typing import NamedTuple

import numpy as np
from pydantic import BaseModel

from exaflow.aggregation_clients import AggregationType
from exaflow.algorithms.exaflow.algorithm import Algorithm
from exaflow.algorithms.exaflow.exaflow_registry import exaflow_udf
from exaflow.algorithms.exaflow.library.linear_models import (
    run_distributed_linear_regression,
)
from exaflow.algorithms.exaflow.metrics import build_design_matrix
from exaflow.algorithms.exaflow.metrics import collect_categorical_levels_from_df
from exaflow.algorithms.exaflow.metrics import construct_design_labels
from exaflow.algorithms.exaflow.metrics import get_dummy_categories
from exaflow.worker_communication import BadUserInput

ALGORITHM_NAME = "linear_regression_cv"
ALPHA = 0.05


class BasicStats(NamedTuple):
    mean: float
    std: float


class CVLinearRegressionResult(BaseModel):
    dependent_var: str
    indep_vars: List[str]
    n_obs: List[int]
    mean_sq_error: BasicStats
    r_squared: BasicStats
    mean_abs_error: BasicStats
    f_stat: BasicStats


class LinearRegressionCVAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata: dict):
        """
        Cross-validated linear regression using exaflow.

        This mirrors the original exaflow LinearRegressionCVAlgorithm, but
        uses exaflow UDFs and secure aggregation.
        """
        if not self.inputdata.y:
            raise BadUserInput("Linear regression CV requires a dependent variable.")
        if not self.inputdata.x:
            raise BadUserInput("Linear regression CV requires at least one covariate.")

        y_var = self.inputdata.y[0]
        n_splits = self.parameters.get("n_splits")
        if not isinstance(n_splits, int) or n_splits <= 1:
            raise BadUserInput(
                "Parameter 'n_splits' must be an integer greater than 1."
            )

        # Identify categorical vs numerical predictors
        categorical_vars = [
            var for var in self.inputdata.x if metadata[var]["is_categorical"]
        ]
        numerical_vars = [
            var for var in self.inputdata.x if not metadata[var]["is_categorical"]
        ]

        # Discover dummy categories from actual data (shared util)
        dummy_categories = get_dummy_categories(
            engine=self.engine,
            inputdata_json=self.inputdata.json(),
            categorical_vars=categorical_vars,
            collect_udf=linear_collect_categorical_levels_cv,
        )

        indep_var_names = construct_design_labels(
            categorical_vars=categorical_vars,
            dummy_categories=dummy_categories,
            numerical_vars=numerical_vars,
        )
        # Number of predictors excluding intercept
        p = len(indep_var_names) - 1

        # 1) Check per-worker that n_obs >= n_splits
        check_results = self.engine.run_algorithm_udf(
            func=linear_regression_cv_check_local,
            positional_args={
                "inputdata": self.inputdata.json(),
                "y_var": y_var,
                "n_splits": n_splits,
            },
        )
        if not all(res["ok"] for res in check_results):
            raise BadUserInput(
                "Cross validation cannot run because some of the workers "
                "participating in the experiment have a number of observations "
                f"smaller than the number of splits, {n_splits}."
            )

        # 2) Run distributed K-fold CV with aggregation server
        udf_results = self.engine.run_algorithm_udf(
            func=linear_regression_cv_local_step,
            positional_args={
                "inputdata": self.inputdata.json(),
                "y_var": y_var,
                "categorical_vars": categorical_vars,
                "numerical_vars": numerical_vars,
                "dummy_categories": dummy_categories,
                "n_splits": n_splits,
                "p": p,
            },
        )

        # All workers should return identical global metrics; take the first
        metrics = udf_results[0]

        rmse = np.asarray(metrics["rmse"], dtype=float)
        r2 = np.asarray(metrics["r2"], dtype=float)
        mae = np.asarray(metrics["mae"], dtype=float)
        fstats = np.asarray(metrics["f_stat"], dtype=float)
        nobs = [int(v) for v in metrics["n_obs"]]

        result = CVLinearRegressionResult(
            dependent_var=y_var,
            indep_vars=indep_var_names,
            n_obs=nobs,
            mean_sq_error=BasicStats(
                mean=float(rmse.mean()), std=float(rmse.std(ddof=1))
            ),
            r_squared=BasicStats(mean=float(r2.mean()), std=float(r2.std(ddof=1))),
            mean_abs_error=BasicStats(
                mean=float(mae.mean()), std=float(mae.std(ddof=1))
            ),
            f_stat=BasicStats(mean=float(fstats.mean()), std=float(fstats.std(ddof=1))),
        )
        return result


# ---------------------------------------------------------------------------
# Helper UDFs
# ---------------------------------------------------------------------------


@exaflow_udf()
def linear_collect_categorical_levels_cv(data, inputdata, categorical_vars):
    """
    Thin UDF wrapper used only to collect categorical levels from workers.

    It delegates the core logic to collect_categorical_levels_from_df so that
    linear/logistic regressions share the same behaviour.
    """

    return collect_categorical_levels_from_df(data, categorical_vars)


@exaflow_udf()
def linear_regression_cv_check_local(data, inputdata, y_var, n_splits):
    """
    Check on each worker whether the number of observations is at least n_splits.
    """

    if y_var in data.columns:
        n_obs = int(data[y_var].dropna().shape[0])
    else:
        n_obs = 0

    return {"ok": bool(n_obs >= n_splits), "n_obs": n_obs}


@exaflow_udf(with_aggregation_server=True)
def linear_regression_cv_local_step(
    data,
    inputdata,
    agg_client,
    y_var,
    categorical_vars,
    numerical_vars,
    dummy_categories,
    n_splits,
    p,
):
    """
    Run K-fold CV locally on each worker, but use agg_client to:

    - Train a global linear model per fold (aggregated X'X, X'y, n_train).
    - Aggregate residual statistics on the test set.

    Returns identical global metrics from every worker.
    """
    from sklearn.model_selection import KFold

    # Ensure n_splits and p are ints
    n_splits = int(n_splits)
    p = int(p)

    if data.empty or y_var not in data.columns:
        # This worker contributes nothing but must still participate.
        return {
            "n_obs": [],
            "rmse": [],
            "r2": [],
            "mae": [],
            "f_stat": [],
        }

    # Build design matrix and target vector
    X = build_design_matrix(
        data,
        categorical_vars=categorical_vars,
        dummy_categories=dummy_categories,
        numerical_vars=numerical_vars,
    )
    y = data[y_var].astype(float).to_numpy().reshape(-1, 1)

    n_rows = X.shape[0]
    if n_rows < n_splits:
        # Should have been caught by the check UDF, but be defensive
        return {
            "n_obs": [],
            "rmse": [],
            "r2": [],
            "mae": [],
            "f_stat": [],
        }

    kf = KFold(n_splits=n_splits, shuffle=False)

    n_obs_per_fold = []
    rmse_per_fold = []
    r2_per_fold = []
    mae_per_fold = []
    fstat_per_fold = []

    for train_idx, test_idx in kf.split(X):
        X_train = X[train_idx, :]
        y_train = y[train_idx, :]
        X_test = X[test_idx, :]
        y_test = y[test_idx, :]

        # --------------------------
        # Training: global OLS model
        # --------------------------
        train_stats = run_distributed_linear_regression(
            agg_client=agg_client,
            X=X_train,
            y=y_train,
        )

        n_train = int(train_stats["n_obs"])
        coeff = np.asarray(train_stats["coefficients"], dtype=float).reshape(-1, 1)

        if n_train == 0 or coeff.size == 0:
            n_obs_per_fold.append(0)
            rmse_per_fold.append(0.0)
            r2_per_fold.append(0.0)
            mae_per_fold.append(0.0)
            fstat_per_fold.append(0.0)
            continue

        # --------------------------
        # Evaluation on test set
        # --------------------------
        if X_test.size == 0:
            n_obs_per_fold.append(n_train)
            rmse_per_fold.append(0.0)
            r2_per_fold.append(0.0)
            mae_per_fold.append(0.0)
            fstat_per_fold.append(0.0)
            continue

        y_pred_local = X_test @ coeff  # (n_test_local, 1)
        resid_local = y_test - y_pred_local

        rss_local = float(np.dot(resid_local.reshape(-1), resid_local.reshape(-1)))
        sum_abs_resid_local = float(np.abs(resid_local).sum())
        n_test_local = float(y_test.shape[0])

        sum_y_test_local = float(y_test.sum())
        sum_sq_y_test_local = float((y_test**2).sum())

        # Aggregate test statistics across workers
        (
            rss_arr,
            sum_abs_resid_arr,
            n_test_arr,
            sum_y_test_arr,
            sum_sq_y_test_arr,
        ) = agg_client.aggregate_batch(
            [
                (AggregationType.SUM, np.array([rss_local], dtype=float)),
                (AggregationType.SUM, np.array([sum_abs_resid_local], dtype=float)),
                (AggregationType.SUM, np.array([n_test_local], dtype=float)),
                (AggregationType.SUM, np.array([sum_y_test_local], dtype=float)),
                (AggregationType.SUM, np.array([sum_sq_y_test_local], dtype=float)),
            ]
        )

        rss = float(np.asarray(rss_arr, dtype=float).reshape(-1)[0])
        sum_abs_resid = float(np.asarray(sum_abs_resid_arr, dtype=float).reshape(-1)[0])
        n_test = int(np.asarray(n_test_arr, dtype=float).reshape(-1)[0])
        sum_y_test = float(np.asarray(sum_y_test_arr, dtype=float).reshape(-1)[0])
        sum_sq_y_test = float(np.asarray(sum_sq_y_test_arr, dtype=float).reshape(-1)[0])

        # Global TSS on test set
        if n_test > 0:
            y_mean_test = sum_y_test / n_test
            tss = (
                sum_sq_y_test
                - 2.0 * y_mean_test * sum_y_test
                + n_test * (y_mean_test**2)
            )
        else:
            tss = 0.0

        # Degrees of freedom: same as original linear regression:
        # df = n_obs_train - p - 1 (p excludes intercept)
        df_resid = n_train - p - 1

        if df_resid <= 0 or n_test == 0 or rss <= 0.0 or tss <= 0.0 or p <= 0:
            r2_val = 0.0
            rmse_val = 0.0
            mae_val = 0.0
            f_val = 0.0
        else:
            r2_val = 1.0 - (rss / tss)
            rmse_val = float(np.sqrt(rss / n_test))
            mae_val = float(sum_abs_resid / n_test)
            f_val = float((tss - rss) * df_resid / (p * rss))

        n_obs_per_fold.append(n_train)
        rmse_per_fold.append(rmse_val)
        r2_per_fold.append(r2_val)
        mae_per_fold.append(mae_val)
        fstat_per_fold.append(f_val)

    return {
        "n_obs": n_obs_per_fold,
        "rmse": rmse_per_fold,
        "r2": r2_per_fold,
        "mae": mae_per_fold,
        "f_stat": fstat_per_fold,
    }
