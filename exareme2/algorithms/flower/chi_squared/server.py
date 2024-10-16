import os
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import flwr as fl
from flwr.common import EvaluateRes
from flwr.common import FitRes
from flwr.common import Parameters
from flwr.common import Scalar
from flwr.common.logger import FLOWER_LOGGER
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from exareme2.algorithms.flower.inputdata_preprocessing import post_result

NUM_OF_ROUNDS = 5
num_clients_per_round = 2
pool_size = 2
num_evaluate_clients = 2


class CustomChiSquaredFedAvg(FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Filter out clients with 0 examples
        valid_results = [
            (client, res) for client, res in results if res.num_examples > 0
        ]

        # If no valid results, return None
        if not valid_results:
            return None, {}

        # Proceed with aggregation using only valid clients
        scaling_factors = [
            fit_res.num_examples / sum(res.num_examples for _, res in valid_results)
            for _, fit_res in valid_results
        ]

        # Aggregate parameters (no model parameters in this case)
        # Custom aggregation logic, etc.

        return Parameters(tensor_type="", tensors=[]), {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}

        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(
                server_round, eval_metrics
            )
        elif server_round == 1:
            FLOWER_LOGGER.warn("No evaluate_metrics_aggregation_fn provided")

        return 0, metrics_aggregated


def evaluate_metrics_aggregation(server_round, eval_metrics):
    total_num = sum([num for num, _ in eval_metrics])
    p_value_aggregated = (
        sum([metrics["p_value"] * num for num, metrics in eval_metrics]) / total_num
    )
    metrics_aggregated = {"p_value": p_value_aggregated}

    if server_round == NUM_OF_ROUNDS:
        post_result({"metrics_aggregated": metrics_aggregated})

    return metrics_aggregated


if __name__ == "__main__":
    strategy = CustomChiSquaredFedAvg(
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
