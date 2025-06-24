from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import grpc

from aggregation_server import aggregation_server_pb2 as pb2
from exareme2.aggregation_client.base_aggregation_client import BaseAggregationClient


class AggregationControllerClient(ControllerAggregationClient):
    """
    gRPC wrapper used *only* by the controller to configure / clean up the
    aggregation_server service.
    """

    # -- life-cycle -------------------------------------------------------- #
    def configure(self, num_workers: int) -> str:
        response = self._stub.Configure(
            pb2.ConfigureRequest(
                request_id=self._request_id, num_of_workers=num_workers
            )
        )
        return response.status

    def cleanup(self) -> str:
        try:
            response = self._stub.Cleanup(
                pb2.CleanupRequest(request_id=self._request_id)
            )
            return response.status
        except grpc._channel._InactiveRpcError:
            # aggregation_server already shut down remotely – not an error
            return "AggregationServer already offline"

    # -- context-manager convenience -------------------------------------- #
    def __enter__(self) -> "AggregationControllerClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
