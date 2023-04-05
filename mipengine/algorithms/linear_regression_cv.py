from typing import List
from typing import NamedTuple

import numpy
from pydantic import BaseModel

from mipengine.algorithm_specification import AlgorithmSpecification
from mipengine.algorithm_specification import InputDataSpecification
from mipengine.algorithm_specification import InputDataSpecifications
from mipengine.algorithm_specification import InputDataStatType
from mipengine.algorithm_specification import InputDataType
from mipengine.algorithm_specification import ParameterSpecification
from mipengine.algorithm_specification import ParameterType
from mipengine.algorithms.algorithm import Algorithm
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


class LinearRegressionCVAlgorithm(Algorithm, algname="linear_regression_cv"):
    @classmethod
    def get_specification(cls):
        return AlgorithmSpecification(
            name=cls.algname,
            desc="Method used to evaluate the performance of a linear regression model. It involves splitting the data into training and validation sets and testing the model's ability to generalize to new data by using the validation set.",
            label="Linear Regression Cross-validation",
            enabled=True,
            inputdata=InputDataSpecifications(
                x=InputDataSpecification(
                    label="Covariates (independent)",
                    desc="One or more variables. Can be numerical or nominal. For nominal variables dummy encoding is used.",
                    types=[InputDataType.REAL, InputDataType.INT, InputDataType.TEXT],
                    stattypes=[InputDataStatType.NUMERICAL, InputDataStatType.NOMINAL],
                    notblank=True,
                    multiple=True,
                ),
                y=InputDataSpecification(
                    label="Variable (dependent)",
                    desc="A unique numerical variable.",
                    types=[InputDataType.REAL],
                    stattypes=[InputDataStatType.NUMERICAL],
                    notblank=True,
                    multiple=False,
                ),
            ),
            parameters={
                "n_splits": ParameterSpecification(
                    label="Number of splits",
                    desc="Number of splits for cross-validation.",
                    types=[ParameterType.INT],
                    notblank=True,
                    multiple=False,
                    default=5,
                    min=2,
                    max=20,
                ),
            },
        )

    def get_variable_groups(self):
        return [self.variables.x, self.variables.y]

    def run(self, engine, data_model_views, metadata):
        X, y = data_model_views

        n_splits = self.algorithm_parameters["n_splits"]

        dummy_encoder = DummyEncoder(engine=engine, metadata=metadata)
        X = dummy_encoder.transform(X)

        p = len(dummy_encoder.new_varnames) - 1

        kf = KFold(engine, n_splits=n_splits)
        X_train, X_test, y_train, y_test = kf.split(X, y)

        models = [LinearRegression(engine) for _ in range(n_splits)]

        for model, X, y in zip(models, X_train, y_train):
            model.fit(X=X, y=y)

        for model, X, y in zip(models, X_test, y_test):
            y_pred = model.predict(X)
            model.compute_summary(
                y_test=relation_to_vector(y, engine),
                y_pred=y_pred,
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
