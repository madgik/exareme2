from typing import List
from typing import NamedTuple

import numpy
from pydantic import BaseModel

from mipengine.algorithms.linear_regression import LinearRegression
from mipengine.algorithms.preprocessing import DummyEncoder
from mipengine.algorithms.preprocessing import KFold
from mipengine.algorithms.preprocessing import relation_to_vector


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


def run(executor):
    x_vars, y_vars = executor.x_variables, executor.y_variables
    X, y = executor.create_primary_data_views(variable_groups=[x_vars, y_vars])
    n_splits = executor.algorithm_parameters["n_splits"]

    dummy_encoder = DummyEncoder(executor)
    X = dummy_encoder.transform(X)

    p = len(dummy_encoder.new_varnames) - 1

    kf = KFold(executor, n_splits=n_splits)
    X_train, X_test, y_train, y_test = kf.split(X, y)

    models = [LinearRegression(executor) for _ in range(n_splits)]

    for model, X, y in zip(models, X_train, y_train):
        model.fit(X=X, y=y)

    for model, X, y in zip(models, X_test, y_test):
        y_pred = model.predict(X)
        model.compute_summary(
            y_test=relation_to_vector(y, executor),
            y_pred=y_pred,
            p=p,
        )

    rms_errors = numpy.array([m.rmse for m in models])
    r2s = numpy.array([m.r_squared for m in models])
    maes = numpy.array([m.mae for m in models])
    f_stats = numpy.array([m.f_stat for m in models])

    result = CVLinearRegressionResult(
        dependent_var=executor.y_variables[0],
        indep_vars=dummy_encoder.new_varnames,
        n_obs=[m.n_obs for m in models],
        mean_sq_error=BasicStats(mean=rms_errors.mean(), std=rms_errors.std(ddof=1)),
        r_squared=BasicStats(mean=r2s.mean(), std=r2s.std(ddof=1)),
        mean_abs_error=BasicStats(mean=maes.mean(), std=maes.std(ddof=1)),
        f_stat=BasicStats(mean=f_stats.mean(), std=f_stats.std(ddof=1)),
    )
    return result
