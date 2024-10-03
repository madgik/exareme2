import os
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import flwr as fl
from flwr.common import EvaluateRes
from flwr.common import Scalar
from flwr.common.logger import FLOWER_LOGGER
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedXgbBagging

from exareme2.algorithms.flower.inputdata_preprocessing import post_result

# FL experimental settings
pool_size = 2
NUM_OF_ROUNDS = 5
num_clients_per_round = 2
num_evaluate_clients = 2


class Custom_fed_xgboost(FedXgbBagging):
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation metrics using average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(
                server_round, eval_metrics
            )
        elif server_round == 1:  # Only log this warning once
            FLOWER_LOGGER.warn("No evaluate_metrics_aggregation_fn provided")

        return 0, metrics_aggregated


def evaluate_metrics_aggregation(server_round, eval_metrics):
    """Return an aggregated metric (AUC) for evaluation."""

    total_num = sum([num for num, _ in eval_metrics])
    auc_aggregated = (
        sum([metrics["AUC"] * num for num, metrics in eval_metrics]) / total_num
    )
    metrics_aggregated = {"AUC": auc_aggregated}
    if server_round == NUM_OF_ROUNDS:
        post_result({"metrics_aggregated": metrics_aggregated})
    return metrics_aggregated


if __name__ == "__main__":
    # Define strategy
    strategy = Custom_fed_xgboost(
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
