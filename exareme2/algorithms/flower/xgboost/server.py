import os

import flwr as fl
from flwr.server.strategy import FedXgbBagging

from exareme2.algorithms.flower.inputdata_preprocessing import post_result

# FL experimental settings
pool_size = 2
NUM_OF_ROUNDS = 5
num_clients_per_round = 2
num_evaluate_clients = 2


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

    def aggregate_evaluate(self, rnd, results, failures):
        aggregated_metrics = super().aggregate_evaluate(rnd, results, failures)
        if rnd == self.num_rounds:
            post_result({"metrics_aggregated": aggregated_metrics})
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
