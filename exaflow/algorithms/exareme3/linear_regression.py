from typing import List

import numpy as np
from pydantic import BaseModel

from exaflow.algorithms.exareme3.algorithm import Algorithm
from exaflow.algorithms.exareme3.exaflow_registry import exaflow_udf
from exaflow.algorithms.exareme3.library.linear_models import compute_summary_from_stats
from exaflow.algorithms.exareme3.library.linear_models import (
    run_distributed_linear_regression,
)
from exaflow.algorithms.exareme3.metadata_utils import validate_metadata_vars
from exaflow.algorithms.exareme3.metrics import build_design_matrix
from exaflow.algorithms.exareme3.metrics import collect_categorical_levels_from_df
from exaflow.algorithms.exareme3.metrics import construct_design_labels
from exaflow.algorithms.exareme3.preprocessing import get_dummy_categories
from exaflow.algorithms.exareme3.validation_utils import require_covariates
from exaflow.algorithms.exareme3.validation_utils import require_dependent_var

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
        require_dependent_var(
            self.inputdata,
            message="Linear regression requires a dependent variable.",
        )
        require_covariates(
            self.inputdata,
            message="Linear regression requires at least one covariate.",
        )

        y_var = self.inputdata.y[0]
        validate_metadata_vars([y_var] + self.inputdata.x, metadata)

        categorical_vars = [
            var for var in self.inputdata.x if metadata[var]["is_categorical"]
        ]
        numerical_vars = [
            var for var in self.inputdata.x if not metadata[var]["is_categorical"]
        ]

        # Discover dummy categories across workers
        dummy_categories = get_dummy_categories(
            engine=self.engine,
            inputdata_json=self.inputdata.json(),
            categorical_vars=categorical_vars,
            collect_udf=linear_collect_categorical_levels,
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
def linear_collect_categorical_levels(data, inputdata, categorical_vars):
    return collect_categorical_levels_from_df(data, categorical_vars)


@exaflow_udf(with_aggregation_server=True)
def linear_regression_local_step(
    data,
    inputdata,
    agg_client,
    y_var,
    categorical_vars,
    numerical_vars,
    dummy_categories,
):

    y = data[y_var].to_numpy(dtype=float, copy=False).reshape(-1, 1)
    X = build_design_matrix(
        data,
        categorical_vars=categorical_vars,
        dummy_categories=dummy_categories,
        numerical_vars=numerical_vars,
    )

    model_stats = run_distributed_linear_regression(agg_client, X, y)
    return model_stats
