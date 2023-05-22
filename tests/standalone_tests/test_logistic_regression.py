import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression as LogisticRegressionSKL

from mipengine.algorithms.logistic_regression import LogisticRegression
from tests.standalone_tests.test_linear_regression import InMemoryExecutionEngine

np.random.seed(1)


@pytest.mark.slow
class TestLogisticRegression:
    @pytest.mark.parametrize("nrows", range(10, 100, 10))
    @pytest.mark.parametrize("ncols", range(1, 20))
    def test_predict(self, nrows, ncols):
        X = pd.DataFrame(np.random.randn(nrows, ncols))
        y = np.random.random_integers(0, 1, size=nrows)
        coef, expected_pred = self._get_sklearn_coef_and_pred(X, y)
        lr = LogisticRegression(engine=InMemoryExecutionEngine())
        lr.coeff = coef

        result_pred = lr.predict_proba(X)

        np.testing.assert_allclose(result_pred["proba"], expected_pred)

    @staticmethod
    def _get_sklearn_coef_and_pred(X, y):
        lr = LogisticRegressionSKL(fit_intercept=False)
        lr.fit(X, y)
        coef = lr.coef_
        expected_pred = lr.predict_proba(X)
        return coef.reshape(-1), expected_pred[:, 1]
