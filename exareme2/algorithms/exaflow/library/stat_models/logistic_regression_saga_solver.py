import numpy as np
from sklearn.linear_model import LogisticRegression

from exareme2.aggregation_clients.exaflow_udf_aggregation_client import (
    ExaflowUDFAggregationClient as AggregationClient,
)
from exareme2.algorithms.exaflow.library.templates.statistical_model import (
    StatisticalModel,
)


class FederatedLogisticRegressionClientSaSo(StatisticalModel):

    def __init__(self, client: AggregationClient, model_params=None, classes=None):

        super().__init__(client)
        self.model_params = model_params or {
            "solver": "saga",
            "penalty": "l2",
            "fit_intercept": True,
            "max_iter": 100,
            "warm_start": True,
        }
        self.model = LogisticRegression(**self.model_params)
        self.classes = np.asarray(classes) if classes is not None else None

    def fit(self, X: np.ndarray, y: np.ndarray, num_epochs: int = 100):
        """
        Federated training loop. Performs one epoch of local training followed
        by federated aggregation after each epoch.

        Args:
            num_epochs (int): Number of global training epochs (rounds of aggregation).
        """
        for epoch in range(num_epochs):

            # One local epoch: partial_fit for better control
            if hasattr(self.model, "partial_fit"):
                if epoch == 0:
                    # Ensure a consistent class order across workers.
                    classes = self.classes if self.classes is not None else np.unique(y)
                    self.model.partial_fit(X, y, classes=classes)
                else:
                    self.model.partial_fit(X, y)
            else:
                # Fallback to full fit, warm_start avoids reinitializing weights
                self.model.fit(X, y)

            # Extract weights (coef_ and intercept_)
            self.model.coef_ = self.client.fed_weighted_avg(
                self.model.coef_, X.shape[0]
            )
            self.model.intercept_ = self.client.fed_weighted_avg(
                self.model.intercept_, X.shape[0]
            )

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)
