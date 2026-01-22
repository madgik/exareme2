import warnings

import flwr as fl
import numpy as np
from flwr.common.logger import FLOWER_LOGGER
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from exaflow.algorithms.flower.inputdata_preprocessing import connect_with_retries
from tests.algorithms.flower.mnist_logistic_regression import utils

if __name__ == "__main__":
    # Load data from file
    X, y = utils.load_data()

    # Split the on edge data: 80% train, 20% test
    X_train, X_test = X[: int(0.8 * len(X))], X[int(0.8 * len(X)) :]
    y_train, y_test = y[: int(0.8 * len(y))], y[int(0.8 * len(y)) :]

    # Create LogisticRegression Model
    model = LogisticRegression(
        penalty="l2",
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )

    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model)

    # Define Flower client
    class MnistClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            # Ensure parameters are returned as a list of NumPy arrays
            return [
                param.astype(np.float32) for param in utils.get_model_parameters(model)
            ]

        def fit(self, parameters, config):
            try:
                utils.set_model_params(model, parameters)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_train, y_train)
                accuracy = model.score(X_test, y_test)

                # Ensure the parameters are extracted and formatted correctly
                params = [
                    param.astype(np.float32)
                    for param in utils.get_model_parameters(model)
                ]
                return_data = (params, len(X_train), {"accuracy": accuracy})
            except Exception as e:
                FLOWER_LOGGER.error(f"Error during model fitting: {e}")
                # On error, default to zero-initialized parameters, no training examples, and zero accuracy
                zero_params = [
                    np.zeros_like(param) for param in utils.get_model_parameters(model)
                ]
                return_data = (zero_params, 0, {"accuracy": 0.0})

            return return_data

        def evaluate(self, parameters, config):
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}

    # MnistClient
    mnist_client = MnistClient()
    connect_with_retries(mnist_client, "MnistClient")
