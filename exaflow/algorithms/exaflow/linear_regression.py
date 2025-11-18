from typing import Dict
from typing import List

import numpy as np
from pydantic import BaseModel
from scipy import stats

from exaflow.algorithms.exaflow.algorithm import Algorithm
from exaflow.algorithms.exaflow.exaflow_registry import exaflow_udf
from exaflow.algorithms.exaflow.metrics import build_design_matrix
from exaflow.algorithms.exaflow.metrics import collect_categorical_levels_from_df
from exaflow.algorithms.exaflow.metrics import construct_design_labels
from exaflow.algorithms.exaflow.metrics import get_dummy_categories
from exaflow.worker_communication import BadUserInput

ALPHA = 0.05  # same role as in exaflow version
ALGORITHM_NAME = "linear_regression"


class LinearRegressionResult(BaseModel):
    dependent_var: str
    n_obs: int
    df_resid: float
    df_model: float
    rse: float
    r_squared: float
    r_squared_adjusted: float
    f_stat: float
    f_pvalue: float
    indep_vars: List[str]
    coefficients: List[float]
    std_err: List[float]
    t_stats: List[float]
    pvalues: List[float]
    lower_ci: List[float]
    upper_ci: List[float]


class LinearRegressionAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata: dict):
        # Basic input checks
        if not self.inputdata.y:
            raise BadUserInput("Linear regression requires a dependent variable.")
        if not self.inputdata.x:
            raise BadUserInput("Linear regression requires at least one covariate.")
        use_duckdb = True

        y_var = self.inputdata.y[0]

        categorical_vars = [
            var for var in self.inputdata.x if metadata[var]["is_categorical"]
        ]
        numerical_vars = [
            var for var in self.inputdata.x if not metadata[var]["is_categorical"]
        ]

        # Discover dummy categories across workers
        # TODO I do not know if i like this its way to confusing
        dummy_categories = get_dummy_categories(
            self.engine,
            self.inputdata.json(),
            categorical_vars,
            linear_collect_categorical_levels,
            extra_args={"use_duckdb": use_duckdb},
        )

        # Construct names of design-matrix columns: Intercept, dummies, numericals
        indep_var_names = construct_design_labels(
            categorical_vars=categorical_vars,
            dummy_categories=dummy_categories,
            numerical_vars=numerical_vars,
        )

        udf_results = self.engine.run_algorithm_udf(
            func=linear_regression_local_step,
            positional_args={
                "inputdata": self.inputdata.json(),
                "y_var": y_var,
                "categorical_vars": categorical_vars,
                "numerical_vars": numerical_vars,
                "dummy_categories": dummy_categories,
                "use_duckdb": use_duckdb,
            },
        )

        model_stats = udf_results[0]

        coefficients = np.array(model_stats["coefficients"], dtype=float)
        xTx_inv = np.array(model_stats["xTx_inv"], dtype=float)
        rss = float(model_stats["rss"])
        tss = float(model_stats["tss"])
        sum_abs_resid = float(model_stats["sum_abs_resid"])
        n_obs = int(model_stats["n_obs"])

        # Number of predictors excluding intercept
        p = len(indep_var_names) - 1

        summary = compute_summary_from_stats(
            coefficients=coefficients,
            xTx_inv=xTx_inv,
            rss=rss,
            tss=tss,
            sum_abs_resid=sum_abs_resid,
            n_obs=n_obs,
            p=p,
        )

        return LinearRegressionResult(
            dependent_var=y_var,
            indep_vars=indep_var_names,
            **summary,
        )


@exaflow_udf()
def linear_collect_categorical_levels(
    inputdata, csv_paths, categorical_vars, use_duckdb=False
):
    from exaflow.algorithms.exaflow.data_loading import load_algorithm_dataframe

    data = load_algorithm_dataframe(inputdata, csv_paths, dropna=True)
    return collect_categorical_levels_from_df(data, categorical_vars)


@exaflow_udf(with_aggregation_server=True)
def linear_regression_local_step(
    inputdata,
    csv_paths,
    agg_client,
    y_var,
    categorical_vars,
    numerical_vars,
    dummy_categories,
    use_duckdb,
):
    from exaflow.algorithms.exaflow.data_loading import load_algorithm_dataframe

    data = load_algorithm_dataframe(inputdata, csv_paths, dropna=True)

    if data.empty:
        X = build_design_matrix(
            data,
            categorical_vars=categorical_vars,
            dummy_categories=dummy_categories,
            numerical_vars=numerical_vars,
        )
        y = np.empty((0, 1), dtype=float)
    else:
        y = data[y_var].astype(float).to_numpy().reshape(-1, 1)
        X = build_design_matrix(
            data,
            categorical_vars=categorical_vars,
            dummy_categories=dummy_categories,
            numerical_vars=numerical_vars,
        )

    model_stats = run_distributed_linear_regression(agg_client, X, y)
    return model_stats


def run_distributed_linear_regression(agg_client, X: np.ndarray, y: np.ndarray):
    """OLS in one exaflow UDF using secure aggregation, similar style to logistic."""
    n_features = X.shape[1]

    # Local sufficient statistics
    xTx_local = X.T @ X if n_features > 0 else np.zeros((0, 0), dtype=float)
    xTy_local = X.T @ y if n_features > 0 else np.zeros((0, 1), dtype=float)
    n_obs_local = float(y.size)
    sum_y_local = float(y.sum())
    sum_sq_y_local = float((y**2).sum())

    # Global (secure) aggregates
    if n_features > 0:
        xTx_flat = agg_client.sum(xTx_local.ravel().tolist())
        xTy_flat = agg_client.sum(xTy_local.ravel().tolist())
        xTx = np.asarray(xTx_flat, dtype=float).reshape((n_features, n_features))
        xTy = np.asarray(xTy_flat, dtype=float).reshape((n_features, 1))
    else:
        xTx = np.zeros((0, 0), dtype=float)
        xTy = np.zeros((0, 1), dtype=float)

    n_obs = int(agg_client.sum([n_obs_local])[0])
    sum_y = float(agg_client.sum([sum_y_local])[0])
    sum_sq_y = float(agg_client.sum([sum_sq_y_local])[0])

    # Solve for coefficients using pseudo-inverse for stability
    if n_features > 0:
        xTx_inv = np.linalg.pinv(xTx)
        coefficients = xTx_inv @ xTy
    else:
        xTx_inv = np.zeros((0, 0), dtype=float)
        coefficients = np.zeros((0, 1), dtype=float)

    # Residuals and RSS need global coefficients.
    # Since agg_client.sum returns the same value to all workers, each worker
    # now has the same global coefficients and can compute local residuals.
    if n_features > 0 and n_obs_local > 0:
        y_pred_local = X @ coefficients
        resid_local = y - y_pred_local
        rss_local = float(np.dot(resid_local.reshape(-1), resid_local.reshape(-1)))
        sum_abs_resid_local = float(np.abs(resid_local).sum())
    else:
        rss_local = 0.0
        sum_abs_resid_local = 0.0

    rss = float(agg_client.sum([rss_local])[0])
    sum_abs_resid = float(agg_client.sum([sum_abs_resid_local])[0])

    # Global TSS from aggregates (same formula as original exaflow)
    if n_obs > 0:
        y_mean = sum_y / n_obs
        tss = sum_sq_y - 2.0 * y_mean * sum_y + n_obs * (y_mean**2)
    else:
        tss = 0.0

    return {
        "coefficients": coefficients.reshape(-1).tolist(),
        "xTx_inv": xTx_inv.tolist(),
        "rss": rss,
        "tss": tss,
        "sum_abs_resid": sum_abs_resid,
        "n_obs": n_obs,
    }


def compute_summary_from_stats(
    *,
    coefficients: np.ndarray,
    xTx_inv: np.ndarray,
    rss: float,
    tss: float,
    sum_abs_resid: float,
    n_obs: int,
    p: int,
) -> dict:
    """
    Replicates the statistics from the original LinearRegression.compute_summary:
    RSE, R², adjusted R², F-stat, t-stats, CIs, etc.
    """
    # Degrees of freedom: same logic as original (p excludes intercept)
    df_resid = n_obs - p - 1
    df_model = p

    if df_resid <= 0 or n_obs <= 0:
        # Degenerate case; avoid divide-by-zero explosions
        return {
            "n_obs": int(n_obs),
            "df_resid": float(df_resid),
            "df_model": float(df_model),
            "rse": float("nan"),
            "r_squared": 0.0,
            "r_squared_adjusted": 0.0,
            "f_stat": float("nan"),
            "f_pvalue": float("nan"),
            "coefficients": coefficients.reshape(-1).tolist(),
            "std_err": [float("nan")] * len(coefficients),
            "t_stats": [float("nan")] * len(coefficients),
            "pvalues": [float("nan")] * len(coefficients),
            "lower_ci": [float("nan")] * len(coefficients),
            "upper_ci": [float("nan")] * len(coefficients),
        }

    # Residual standard error
    rse = float(np.sqrt(rss / df_resid))

    # Standard errors of coefficients: sqrt(σ² * diag((X'X)^-1))
    diag = np.diag(xTx_inv) if xTx_inv.size else np.array([], dtype=float)
    std_err = np.sqrt(np.maximum(diag, 0.0) * (rse**2))

    coeff_flat = coefficients.reshape(-1)
    # t-stats
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.divide(
            coeff_flat,
            std_err,
            out=np.zeros_like(coeff_flat, dtype=float),
            where=std_err != 0,
        )

    # p-values (two-sided)
    t_p_values = stats.t.sf(np.abs(t_stats), df=df_resid) * 2.0

    # Confidence intervals
    t_crit = stats.t.ppf(1.0 - ALPHA / 2.0, df=df_resid)
    lower_ci = (coeff_flat - t_crit * std_err).tolist()
    upper_ci = (coeff_flat + t_crit * std_err).tolist()

    # R² and adjusted R²
    if tss > 0:
        r_squared = 1.0 - (rss / tss)
    else:
        r_squared = 0.0
    r_squared_adjusted = 1.0 - (1.0 - r_squared) * (n_obs - 1) / df_resid

    # F-statistic
    if rss == 0.0 or p == 0:
        f_stat = float("inf")
        f_pvalue = 0.0
    else:
        f_stat = float((tss - rss) * df_resid / (p * rss))
        f_pvalue = float(stats.f.sf(f_stat, dfn=p, dfd=df_resid))

    return {
        "n_obs": int(n_obs),
        "df_resid": float(df_resid),
        "df_model": float(df_model),
        "rse": rse,
        "r_squared": float(r_squared),
        "r_squared_adjusted": float(r_squared_adjusted),
        "f_stat": f_stat,
        "f_pvalue": f_pvalue,
        "coefficients": coeff_flat.tolist(),
        "std_err": std_err.tolist(),
        "t_stats": t_stats.tolist(),
        "pvalues": t_p_values.tolist(),
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
    }
