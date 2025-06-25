from abc import ABC
from abc import abstractmethod
from typing import Optional

from exareme2.controller.services import WorkerLandscapeAggregator
from exareme2.controller.services.api.algorithm_request_dtos import AlgorithmRequestDTO
from exareme2.controller.services.strategy_interface import AlgorithmExecutionStrategyI


class ControllerI(ABC):
    worker_landscape_aggregator: WorkerLandscapeAggregator
    task_timeout: int

    def __init__(
        self,
        worker_landscape_aggregator: WorkerLandscapeAggregator,
        task_timeout: int,
    ) -> None:
        self.worker_landscape_aggregator = worker_landscape_aggregator
        self.task_timeout = task_timeout

    @abstractmethod
    async def exec_algorithm(
        self,
        algorithm_name: str,
        algorithm_request_dto: AlgorithmRequestDTO,
        strategy: Optional[AlgorithmExecutionStrategyI],
    ):
        pass
