from typing import Dict
from typing import List

import numpy as np
from pydantic import BaseModel
from scipy import stats
from scipy.special import expit
from scipy.special import xlogy

from exareme2.algorithms.exaflow.algorithm import Algorithm
from exareme2.algorithms.exaflow.exaflow_registry import exaflow_udf
from exareme2.worker_communication import BadUserInput

ALGORITHM_NAME = "logistic_regression_exaflow_aggregator"
MAX_ITER = 50
TOL = 1e-4
ALPHA = 0.05


class LogisticRegressionSummary(BaseModel):
    n_obs: int
    coefficients: List[float]
    stderr: List[float]
    lower_ci: List[float]
    upper_ci: List[float]
    z_scores: List[float]
    pvalues: List[float]
    df_model: int
    df_resid: int
    r_squared_cs: float
    r_squared_mcf: float
    ll0: float
    ll: float
    aic: float
    bic: float


class LogisticRegressionResult(BaseModel):
    dependent_var: str
    indep_vars: List[str]
    summary: LogisticRegressionSummary


class LogisticRegressionAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata: dict):
        if not self.inputdata.y:
            raise BadUserInput("Logistic regression requires a dependent variable.")
        if not self.inputdata.x:
            raise BadUserInput("Logistic regression requires at least one covariate.")

        positive_class = self.parameters.get("positive_class")
        if positive_class is None:
            raise BadUserInput("Parameter 'positive_class' is required.")

        y_var = self.inputdata.y[0]
        categorical_vars = [
            var for var in self.inputdata.x if metadata[var]["is_categorical"]
        ]
        numerical_vars = [
            var for var in self.inputdata.x if not metadata[var]["is_categorical"]
        ]

        dummy_categories = self._get_dummy_categories(categorical_vars)
        indep_var_names = construct_design_labels(
            categorical_vars, dummy_categories, numerical_vars
        )

        udf_results = self.engine.run_algorithm_udf(
            func=logistic_regression_local_step,
            positional_args={
                "inputdata": self.inputdata.json(),
                "positive_class": positive_class,
                "y_var": y_var,
                "categorical_vars": categorical_vars,
                "numerical_vars": numerical_vars,
                "dummy_categories": dummy_categories,
            },
        )

        model_stats = udf_results[0]
        summary = compute_summary(
            coefficients=np.array(model_stats["coefficients"], dtype=float),
            h_inv=np.array(model_stats["hessian_inverse"], dtype=float),
            ll=model_stats["ll"],
            n_obs=model_stats["n_obs"],
            y_sum=model_stats["y_sum"],
        )

        return LogisticRegressionResult(
            dependent_var=y_var,
            indep_vars=indep_var_names,
            summary=summary,
        )

    def _get_dummy_categories(self, categorical_vars: List[str]) -> Dict[str, List]:
        if not categorical_vars:
            return {}

        worker_levels = self.engine.run_algorithm_udf(
            func=collect_categorical_levels,
            positional_args={
                "inputdata": self.inputdata.json(),
                "categorical_vars": categorical_vars,
            },
        )

        merged = {var: set() for var in categorical_vars}
        for worker_result in worker_levels:
            for var, levels in worker_result.items():
                merged[var].update(level for level in levels if level is not None)

        sorted_levels = {
            var: sorted(merged.get(var, set())) for var in categorical_vars
        }

        # Drop the first level per categorical variable to avoid multicollinearity.
        return {var: levels[1:] for var, levels in sorted_levels.items()}


def construct_design_labels(
    categorical_vars: List[str],
    dummy_categories: Dict[str, List],
    numerical_vars: List[str],
) -> List[str]:
    labels = ["Intercept"]
    for var in categorical_vars:
        labels.extend([f"{var}[{lvl}]" for lvl in dummy_categories.get(var, [])])
    labels.extend(numerical_vars)
    return labels


@exaflow_udf()
def collect_categorical_levels(inputdata, csv_paths, categorical_vars):
    from exareme2.algorithms.utils.inputdata_utils import fetch_data

    if not categorical_vars:
        return {}

    data = fetch_data(inputdata, csv_paths)
    levels = {}
    for var in categorical_vars:
        if var not in data.columns:
            levels[var] = []
            continue
        values = data[var].dropna().unique().tolist()
        levels[var] = values
    return levels


@exaflow_udf(with_aggregation_server=True)
def logistic_regression_local_step(
    inputdata,
    csv_paths,
    agg_client,
    positive_class,
    y_var,
    categorical_vars,
    numerical_vars,
    dummy_categories,
):
    from exareme2.algorithms.utils.inputdata_utils import fetch_data

    data = fetch_data(inputdata, csv_paths)
    if data.empty:
        X = build_design_matrix(
            data,
            categorical_vars=categorical_vars,
            dummy_categories=dummy_categories,
            numerical_vars=numerical_vars,
        )
        y = np.empty((0, 1), dtype=float)
    else:
        y = (data[y_var] == positive_class).astype(float).to_numpy().reshape(-1, 1)
        X = build_design_matrix(
            data,
            categorical_vars=categorical_vars,
            dummy_categories=dummy_categories,
            numerical_vars=numerical_vars,
        )

    model_stats = run_distributed_logistic_regression(agg_client, X, y)
    return model_stats


def build_design_matrix(
    data,
    *,
    categorical_vars: List[str],
    dummy_categories: Dict[str, List],
    numerical_vars: List[str],
) -> np.ndarray:
    n_rows = len(data)
    columns = [np.ones((n_rows, 1), dtype=float)]

    for var in categorical_vars:
        categories = dummy_categories.get(var, [])
        if var not in data.columns:
            columns.extend([np.zeros((n_rows, 1), dtype=float) for _ in categories])
            continue
        values = data[var]
        for category in categories:
            encoded = (values == category).astype(float).to_numpy().reshape(-1, 1)
            columns.append(encoded)

    for var in numerical_vars:
        if var not in data.columns:
            columns.append(np.zeros((n_rows, 1), dtype=float))
            continue
        num_col = data[var].astype(float).to_numpy().reshape(-1, 1)
        columns.append(num_col)

    return np.hstack(columns) if columns else np.empty((n_rows, 0), dtype=float)


def run_distributed_logistic_regression(agg_client, X: np.ndarray, y: np.ndarray):
    if X.ndim != 2:
        X = np.atleast_2d(X)
    if y.ndim == 2 and y.shape[1] != 1:
        y = y.reshape(-1, 1)
    if X.shape[0] != y.shape[0]:
        if X.shape[1] == y.shape[0]:
            X = X.T
        else:
            raise BadUserInput(
                "Design matrix row count does not match target size for logistic regression."
            )

    n_obs_local = int(y.size)
    y_sum_local = float(y.sum())

    total_n_obs = int(agg_client.sum([float(n_obs_local)])[0])
    total_y_sum = float(agg_client.sum([float(y_sum_local)])[0])

    n_features = X.shape[1]
    handle_logreg_errors(total_n_obs, n_features, total_y_sum)

    coeff = np.zeros((n_features, 1), dtype=float)
    H_inv = np.eye(n_features, dtype=float)
    ll = 0.0

    for _ in range(MAX_ITER):
        eta = X @ coeff
        mu = expit(eta)
        w = mu * (1.0 - mu)

        grad_local = np.einsum("ji,j->i", X, (y - mu).reshape(-1))
        H_local = np.einsum("ji,j,jk->ik", X, w.reshape(-1), X)
        ll_local = np.sum(xlogy(y, mu) + xlogy(1 - y, 1 - mu))

        grad = np.asarray(agg_client.sum(grad_local.tolist()), dtype=float)
        H_flat = agg_client.sum(H_local.ravel().tolist())
        H = np.asarray(H_flat, dtype=float).reshape((n_features, n_features))
        ll = float(agg_client.sum([float(ll_local)])[0])

        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            H_inv = np.linalg.pinv(H)

        coeff = coeff + H_inv @ grad.reshape(-1, 1)

        if max_abs(grad) <= TOL:
            break
    else:
        raise BadUserInput("Logistic regression cannot converge. Cancelling run.")

    return {
        "coefficients": coeff.reshape(-1).tolist(),
        "hessian_inverse": H_inv.tolist(),
        "ll": ll,
        "n_obs": total_n_obs,
        "y_sum": total_y_sum,
    }


def compute_summary(
    *,
    coefficients: np.ndarray,
    h_inv: np.ndarray,
    ll: float,
    n_obs: int,
    y_sum: float,
) -> LogisticRegressionSummary:
    stderr = np.sqrt(np.diag(h_inv))
    z_scores = np.divide(
        coefficients, stderr, out=np.zeros_like(coefficients), where=stderr != 0
    )
    pvalues = stats.norm.sf(np.abs(z_scores)) * 2

    z_crit = stats.norm.ppf(1 - ALPHA / 2)
    lower_ci = (coefficients - z_crit * stderr).tolist()
    upper_ci = (coefficients + z_crit * stderr).tolist()

    df_model = len(coefficients) - 1
    df_resid = n_obs - len(coefficients)

    y_mean = y_sum / n_obs if n_obs else 0
    ll0 = float(xlogy(y_sum, y_mean) + xlogy(n_obs - y_sum, 1.0 - y_mean))

    aic = 2 * len(coefficients) - 2 * ll
    bic = np.log(n_obs) * len(coefficients) - 2 * ll if n_obs else float("inf")

    if np.isclose(ll, 0.0) and np.isclose(ll0, 0.0):
        r2_mcf = 1.0
    else:
        r2_mcf = 1.0 - (ll / ll0)
    r2_cs = 1.0 - np.exp(2.0 * (ll0 - ll) / n_obs) if n_obs else 0.0

    return LogisticRegressionSummary(
        n_obs=int(n_obs),
        coefficients=coefficients.tolist(),
        stderr=stderr.tolist(),
        lower_ci=lower_ci,
        upper_ci=upper_ci,
        z_scores=z_scores.tolist(),
        pvalues=pvalues.tolist(),
        df_model=int(df_model),
        df_resid=int(df_resid),
        r_squared_cs=r2_cs,
        r_squared_mcf=r2_mcf,
        ll0=ll0,
        ll=float(ll),
        aic=aic,
        bic=bic,
    )


def handle_logreg_errors(nobs, p, y_sum):
    if nobs <= p:
        msg = (
            "Logistic regression cannot run because the number of "
            "observarions is smaller than the number of predictors. Please "
            "add mode predictors or select more observations."
        )
        raise BadUserInput(msg)
    if min(y_sum, nobs - y_sum) <= p:
        msg = (
            "Logistic regression cannot run because the number of "
            "observarions in one category is smaller than the number of "
            "predictors. Please add mode predictors or select more "
            "observations for the category in question."
        )
        raise BadUserInput(msg)


def max_abs(values):
    return float(np.max(np.abs(values))) if len(values) else 0.0
