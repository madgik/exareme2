from typing import List

import numpy as np
from pydantic import BaseModel

from exaflow.algorithms.exareme3.algorithm import Algorithm
from exaflow.algorithms.exareme3.exareme3_registry import exareme3_udf
from exaflow.algorithms.exareme3.library.logistic_common import coerce_positive_class
from exaflow.algorithms.exareme3.library.logistic_common import compute_logistic_summary
from exaflow.algorithms.exareme3.library.logistic_common import (
    run_distributed_logistic_regression,
)
from exaflow.algorithms.exareme3.metadata_utils import validate_metadata_vars
from exaflow.algorithms.exareme3.metrics import build_design_matrix
from exaflow.algorithms.exareme3.metrics import collect_categorical_levels_from_df
from exaflow.algorithms.exareme3.metrics import construct_design_labels
from exaflow.algorithms.exareme3.preprocessing import get_dummy_categories
from exaflow.algorithms.exareme3.validation_utils import require_covariates
from exaflow.algorithms.exareme3.validation_utils import require_dependent_var
from exaflow.worker_communication import BadUserInput

ALGORITHM_NAME = "logistic_regression"
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
        require_dependent_var(
            self.inputdata, message="Logistic regression requires a dependent variable."
        )
        require_covariates(
            self.inputdata,
            message="Logistic regression requires at least one covariate.",
        )

        positive_class = self.parameters.get("positive_class")
        if positive_class is None:
            raise BadUserInput("Parameter 'positive_class' is required.")

        y_var = self.inputdata.y[0]
        validate_metadata_vars([y_var] + self.inputdata.x, metadata)
        categorical_vars = [
            var for var in self.inputdata.x if metadata[var]["is_categorical"]
        ]
        numerical_vars = [
            var for var in self.inputdata.x if not metadata[var]["is_categorical"]
        ]

        # Discover dummies from actual data (not metadata)
        dummy_categories = get_dummy_categories(
            engine=self.engine,
            inputdata_json=self.inputdata.json(),
            categorical_vars=categorical_vars,
            collect_udf=logistic_collect_categorical_levels,
        )

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
        summary = LogisticRegressionSummary(
            **compute_logistic_summary(
                coefficients=np.array(model_stats["coefficients"], dtype=float),
                h_inv=np.array(model_stats["hessian_inverse"], dtype=float),
                ll=model_stats["ll"],
                n_obs=model_stats["n_obs"],
                y_sum=model_stats["y_sum"],
                alpha=ALPHA,
            )
        )

        return LogisticRegressionResult(
            dependent_var=y_var,
            indep_vars=indep_var_names,
            summary=summary,
        )


@exareme3_udf()
def logistic_collect_categorical_levels(data, inputdata, categorical_vars):

    return collect_categorical_levels_from_df(data, categorical_vars)


@exareme3_udf(with_aggregation_server=True)
def logistic_regression_local_step(
    data,
    inputdata,
    agg_client,
    positive_class,
    y_var,
    categorical_vars,
    numerical_vars,
    dummy_categories,
):
    # --- keep only the variables we actually use (X + y) and ensure unique names ---
    # order: categorical, numerical, then y
    cols = list(dict.fromkeys(list(categorical_vars) + list(numerical_vars) + [y_var]))

    # subset without deep copies; drop duplicated column names, keeping the first occurrence
    data = data.loc[:, cols]

    # y_var is now guaranteed to be a single 1D column, not a 2D frame
    positive_class = coerce_positive_class(data[y_var], positive_class)
    y = data[y_var].eq(positive_class).to_numpy(dtype=float, copy=False).reshape(-1, 1)

    X = build_design_matrix(
        data,
        categorical_vars=categorical_vars,
        dummy_categories=dummy_categories,
        numerical_vars=numerical_vars,
    )

    model_stats = run_distributed_logistic_regression(agg_client, X, y)
    return model_stats
