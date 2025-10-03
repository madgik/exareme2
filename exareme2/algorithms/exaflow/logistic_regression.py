from typing import List

from pydantic import BaseModel

from exareme2.algorithms.exaflow.algorithm import Algorithm
from exareme2.algorithms.exaflow.exaflow_registry import exaflow_udf
from exareme2.algorithms.exaflow.library.stat_models.logistic_regression_saga_solver import (
    FederatedLogisticRegressionClientSaSo,
)
from exareme2.algorithms.specifications import AlgorithmName

ALGORITHM_NAME = AlgorithmName.EXAFLOW_LOGISTIC_REGRESSION


class LogisticRegressionSummary(BaseModel):
    intercept_: float
    coef_: List[float]


class LogisticRegressionResult(BaseModel):
    dependent_var: str
    indep_vars: List[str]
    summary: LogisticRegressionSummary


class LogisticRegressionAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata):
        positive_class = self.parameters.get("positive_class")
        if positive_class is None:
            raise ValueError("Parameter 'positive_class' is required.")

        results = self.engine.run_algorithm_udf(
            func=local_step,
            positional_args={
                "inputdata": self.inputdata.json(),
                "positive_class": positive_class,
            },
        )
        lr_result = results[0]
        if any(r != lr_result for r in results[1:]):
            raise ValueError("Worker results do not match")

        return LogisticRegressionResult(
            dependent_var=self.inputdata.y[0],
            indep_vars=self.inputdata.x,
            summary=LogisticRegressionSummary.parse_obj(lr_result),
        )


@exaflow_udf(with_aggregation_server=True)
def local_step(inputdata, csv_paths, agg_client, positive_class):
    from exareme2.algorithms.utils.inputdata_utils import fetch_data

    data = fetch_data(inputdata, csv_paths)
    model = FederatedLogisticRegressionClientSaSo(agg_client, classes=[0, 1])

    x = data[inputdata.x].to_numpy()
    y_series = data[inputdata.y[0]] if inputdata.y else None

    if y_series is None:
        raise ValueError("No dependent variable provided in input data.")

    non_null_y = y_series.dropna()
    sample_value = non_null_y.iloc[0] if not non_null_y.empty else None
    positive_class_casted = positive_class
    if sample_value is not None and not isinstance(positive_class, type(sample_value)):
        sample_type = type(sample_value)
        try:
            positive_class_casted = sample_type(positive_class)
        except (TypeError, ValueError):
            positive_class_casted = positive_class

    y = (y_series == positive_class_casted).astype(int).to_numpy().ravel()

    model.fit(x, y, num_epochs=200)

    return {
        "coef_": model.model.coef_.ravel().astype(float).tolist(),
        "intercept_": float(model.model.intercept_.ravel()[0]),
    }
