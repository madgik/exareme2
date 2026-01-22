import copy
import os

import flwr as fl
from flwr.common.logger import FLOWER_LOGGER
from flwr.server.strategy import FedXgbBagging

from exaflow.algorithms.flower.inputdata_preprocessing import post_result

# FL experimental settings
NUM_CLIENTS = int(os.environ["NUMBER_OF_CLIENTS"])
pool_size = NUM_CLIENTS
NUM_OF_ROUNDS = 5
num_clients_per_round = NUM_CLIENTS
num_evaluate_clients = NUM_CLIENTS


def evaluate_metrics_aggregation(eval_metrics):
    """Return an aggregated metric (AUC) for evaluation."""
    total_num = sum([num for num, _ in eval_metrics])
    auc_aggregated = (
        sum([metrics["AUC"] * num for num, metrics in eval_metrics]) / total_num
    )
    metrics_aggregated = {"AUC": auc_aggregated}
    return metrics_aggregated


class CustomFedXgbBagging(FedXgbBagging):
    def __init__(self, num_rounds, **kwargs):
        super().__init__(**kwargs)
        self.num_rounds = num_rounds
        self.initial_auc = 0.0

    def aggregate_evaluate(self, rnd, results, failures):
        aggregated_metrics = super().aggregate_evaluate(rnd, results, failures)
        d2 = copy.deepcopy(aggregated_metrics)
        curr_auc = d2[1]["AUC"]

        if rnd == 1:
            # print(aggregated_metrics)
            d3 = copy.deepcopy(aggregated_metrics)
            curr_auc = d3[1]["AUC"]
            self.initial_auc = curr_auc

        if rnd == self.num_rounds:
            FLOWER_LOGGER.debug("aggregated metrics is " + str(aggregated_metrics))

            auc_diff = curr_auc - self.initial_auc
            auc_ascending = ""
            if auc_diff >= -0.15:
                auc_ascending = "correct"
            else:
                auc_ascending = "not_correct"

            post_result(
                {
                    "AUC": curr_auc,
                    "auc_ascending": auc_ascending,
                    "initial_auc": self.initial_auc,
                }
            )
        return aggregated_metrics


if __name__ == "__main__":
    # Define strategy
    strategy = CustomFedXgbBagging(
        num_rounds=NUM_OF_ROUNDS,
        fraction_fit=(float(num_clients_per_round) / pool_size),
        min_fit_clients=num_clients_per_round,
        min_available_clients=pool_size,
        min_evaluate_clients=num_evaluate_clients,
        fraction_evaluate=1.0,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
    )

    fl.server.start_server(
        server_address=os.environ["SERVER_ADDRESS"],
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=NUM_OF_ROUNDS),
    )
