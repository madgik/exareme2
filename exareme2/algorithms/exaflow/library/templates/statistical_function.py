from abc import ABC
from abc import abstractmethod

from exareme2.aggregation_clients.exaflow_udf_aggregation_client import (
    ExaflowUDFAggregationClient as AggregationClient,
)


class StatisticalFunction(ABC):

    def __init__(self, client: AggregationClient):
        self.client = client

    @abstractmethod
    def compute(self, *args, **kwargs):
        pass
