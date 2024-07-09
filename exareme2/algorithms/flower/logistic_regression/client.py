import os
import time
import warnings
from math import log2

import flwr as fl
from flwr.common.logger import FLOWER_LOGGER
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from utils import get_model_parameters
from utils import set_initial_params
from utils import set_model_params

from exareme2.algorithms.flower.inputdata_preprocessing import fetch_client_data
from exareme2.algorithms.flower.inputdata_preprocessing import get_input
from exareme2.algorithms.flower.inputdata_preprocessing import preprocess_data


class LogisticRegressionClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train

    def get_parameters(self, **kwargs):  # Now accepts any keyword arguments
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        set_model_params(self.model, parameters)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)
        return get_model_parameters(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config):
        set_model_params(self.model, parameters)
        loss = log_loss(self.y_train, self.model.predict_proba(self.X_train))
        accuracy = self.model.score(self.X_train, self.y_train)
        return loss, len(self.X_train), {"accuracy": accuracy}


if __name__ == "__main__":
    model = LogisticRegression(penalty="l2", max_iter=1, warm_start=True)
    inputdata = get_input()
    full_data = fetch_client_data(inputdata)
    X_train, y_train = preprocess_data(inputdata, full_data)
    set_initial_params(model, X_train, full_data, inputdata)

    client = LogisticRegressionClient(model, X_train, y_train)

    attempts = 0
    max_attempts = int(log2(int(os.environ["TIMEOUT"])))
    while True:
        try:
            fl.client.start_client(
                server_address=os.environ["SERVER_ADDRESS"], client=client.to_client()
            )
            FLOWER_LOGGER.debug("Connection successful on attempt", attempts + 1)
            break
        except Exception as e:
            FLOWER_LOGGER.warning(
                f"Connection with the server failed. Attempt {attempts + 1} failed: {e}"
            )
            time.sleep(pow(2, attempts))
            attempts += 1
            if attempts >= max_attempts:
                FLOWER_LOGGER.error("Could not establish connection to the server.")
                raise e
