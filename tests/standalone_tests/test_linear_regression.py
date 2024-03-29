import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression as LinearRegressionSKL

from exareme2.algorithms.exareme2.linear_regression import LinearRegression

np.random.seed(0)


class InMemoryExecutionEngine:
    def run_udf_on_local_nodes(self, func, keyword_args, *args, **kwargs):
        return func(**keyword_args)

    run_udf_on_global_node = run_udf_on_local_nodes


class TestLinearRegression:
    @pytest.mark.parametrize("nrows", range(10, 100, 10))
    @pytest.mark.parametrize("ncols", range(1, 20))
    def test_predict(self, nrows, ncols):
        X = pd.DataFrame(np.random.randn(nrows, ncols))
        y = np.random.randn(nrows)
        coef, expected_pred = self._get_sklearn_coef_and_pred(X, y)
        lr = LinearRegression(engine=InMemoryExecutionEngine())
        lr.coefficients = coef

        result_pred = lr.predict(X)

        np.testing.assert_allclose(result_pred, expected_pred)

    @staticmethod
    def _get_sklearn_coef_and_pred(X, y):
        lr = LinearRegressionSKL(fit_intercept=False)
        lr.fit(X, y)
        coef = lr.coef_
        expected_pred = lr.predict(X)
        return coef, expected_pred
