from typing import List
from typing import NamedTuple

import numpy
from pydantic import BaseModel

from exareme2.algorithms.exareme2.algorithm import Algorithm
from exareme2.algorithms.exareme2.algorithm import AlgorithmDataLoader
from exareme2.algorithms.exareme2.crossvalidation import KFold
from exareme2.algorithms.exareme2.crossvalidation import cross_validate
from exareme2.algorithms.exareme2.linear_regression import LinearRegression
from exareme2.algorithms.exareme2.preprocessing import DummyEncoder
from exareme2.algorithms.exareme2.preprocessing import relation_to_vector
from exareme2.algorithms.specifications import AlgorithmName

ALGORITHM_NAME = AlgorithmName.LINEAR_REGRESSION_CV


class LinearRegressionCVDataLoader(AlgorithmDataLoader, algname=ALGORITHM_NAME):
    def get_variable_groups(self):
        return [self._variables.x, self._variables.y]


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


class LinearRegressionCVAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, data, metadata):
        X, y = data

        n_splits = self.algorithm_parameters["n_splits"]

        dummy_encoder = DummyEncoder(engine=self.engine, metadata=metadata)
        X = dummy_encoder.transform(X)

        p = len(dummy_encoder.new_varnames) - 1

        # Perform cross-validation
        kf = KFold(self.engine, n_splits=n_splits)
        models = [LinearRegression(self.engine) for _ in range(n_splits)]
        y_pred, y_true = cross_validate(X, y, models, kf, pred_type="values")

        for model, y_p, y_t in zip(models, y_pred, y_true):
            model.compute_summary(
                y_test=relation_to_vector(y_t, self.engine),
                y_pred=y_p,
                p=p,
            )

        rms_errors = numpy.array([m.rmse for m in models])
        r2s = numpy.array([m.r_squared for m in models])
        maes = numpy.array([m.mae for m in models])
        f_stats = numpy.array([m.f_stat for m in models])

        result = CVLinearRegressionResult(
            dependent_var=self.variables.y[0],
            indep_vars=dummy_encoder.new_varnames,
            n_obs=[m.n_obs for m in models],
            mean_sq_error=BasicStats(
                mean=rms_errors.mean(), std=rms_errors.std(ddof=1)
            ),
            r_squared=BasicStats(mean=r2s.mean(), std=r2s.std(ddof=1)),
            mean_abs_error=BasicStats(mean=maes.mean(), std=maes.std(ddof=1)),
            f_stat=BasicStats(mean=f_stats.mean(), std=f_stats.std(ddof=1)),
        )
        return result
