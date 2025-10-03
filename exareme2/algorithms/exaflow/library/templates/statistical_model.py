from abc import ABC
from abc import abstractmethod

from exareme2.aggregation_clients.exaflow_udf_aggregation_client import (
    ExaflowUDFAggregationClient as AggregationClient,
)


class StatisticalModel(ABC):

    def __init__(self, client: AggregationClient):
        self.client = client

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass
