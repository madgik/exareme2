from __future__ import annotations

from exareme2.aggregation_clients.controller_aggregation_client import (
    ControllerAggregationClient,
)
from exareme2.controller import logger as ctrl_logger
from exareme2.controller.services.exaflow.exaflow_controller import ExaflowController


class AggregationServerExaflowController(ExaflowController):
    def _configure_aggregator(
        self, request_id: str, workers_info: list[object]
    ) -> object:
        # TODO Remove the direct dependency from the aggregation clients package
        agg_client = ControllerAggregationClient(request_id)
        status = agg_client.configure(num_workers=len(workers_info))
        if status != "Configured":
            raise RuntimeError(f"AggregationServer refused to configure: {status}")
        logger = ExaflowController.__init__.__globals__[
            "ctrl_logger"
        ].get_request_logger(request_id)
        logger.debug(f"AggregationServer configured: {status}")
        return agg_client

    def _cleanup_aggregator(self, agg_client: object) -> None:
        # Clean up aggregation server
        logger = ctrl_logger.get_request_logger(agg_client.request_id)
        cleanup_status = agg_client.cleanup()
        logger.debug(f"AggregationServer cleanup response: {cleanup_status}")
