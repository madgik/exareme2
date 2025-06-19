from __future__ import annotations

from exareme2.controller import logger as ctrl_logger
from exareme2.controller.services.exaflow.exaflow_controller import ExaflowController


class AggregationServerExaflowController(ExaflowController):
    """
    Controller that integrates with an external aggregation server.
    """

    def _configure_aggregator(
        self, request_id: str, workers_info: list[object]
    ) -> object:
        # Lazy import to avoid loading aggregation code when unused
        from exareme2.controller.services.exaflow.aggregation_client import (
            AggregationControllerClient as AggregationClient,
        )

        agg_client = AggregationClient(request_id)
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
