import os

import flwr as fl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from utils import set_initial_params
from utils import set_model_params

from exareme2.algorithms.flower.inputdata_preprocessing import get_input
from exareme2.algorithms.flower.inputdata_preprocessing import post_result
from exareme2.algorithms.flower.inputdata_preprocessing import preprocess_data
from exareme2.algorithms.utils.inputdata_utils import fetch_data

# TODO: NUM_OF_ROUNDS should become a parameter of the algorithm and be set on the AlgorithmRequestDTO
NUM_OF_ROUNDS = 5


def fit_round(server_round: int):
    """Configures the next round of training."""
    return {"server_round": server_round}


def get_evaluate_fn(model, X_test, y_test):
    def evaluate(server_round, parameters, config):
        set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        if server_round == NUM_OF_ROUNDS:
            post_result({"accuracy": accuracy})
        return loss, {"accuracy": accuracy}

    return evaluate


if __name__ == "__main__":
    model = LogisticRegression()
    inputdata = get_input()
    full_data = fetch_data(inputdata, os.getenv("CSV_PATHS").split(","))
    X_train, y_train = preprocess_data(inputdata, full_data)
    set_initial_params(model, X_train, full_data, inputdata)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=int(os.environ["NUMBER_OF_CLIENTS"]),
        evaluate_fn=get_evaluate_fn(model, X_train, y_train),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server(
        server_address=os.environ["SERVER_ADDRESS"],
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=NUM_OF_ROUNDS),
    )
