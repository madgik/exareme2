import os
from typing import Dict

import flwr as fl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from exaflow.algorithms.flower.inputdata_preprocessing import post_result
from tests.algorithms.flower.mnist_logistic_regression import utils

NUM_OF_ROUNDS = 5


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""
    # Load data from file
    X_test, y_test = utils.load_data()

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        if server_round == NUM_OF_ROUNDS:
            post_result({"accuracy": accuracy})
        return loss, {"accuracy": accuracy}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LogisticRegression()
    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=int(os.environ["NUMBER_OF_CLIENTS"]),
        min_evaluate_clients=int(os.environ["NUMBER_OF_CLIENTS"]),
        min_fit_clients=int(os.environ["NUMBER_OF_CLIENTS"]),
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server(
        server_address=os.environ["SERVER_ADDRESS"],
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=NUM_OF_ROUNDS),
    )
