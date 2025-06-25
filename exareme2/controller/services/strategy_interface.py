from abc import ABC
from abc import abstractmethod

from exareme2.controller.services.api.algorithm_request_dtos import AlgorithmRequestDTO
from exareme2.controller.services.controller_interface import ControllerI


class AlgorithmExecutionStrategyI(ABC):
    controller: ControllerI
    algorithm_name: str
    algorithm_request_dto: AlgorithmRequestDTO

    def __init__(
        self,
        controller: ControllerI,
        algorithm_name: str,
        algorithm_request_dto: AlgorithmRequestDTO,
    ):
        self.controller = controller
        self.algorithm_name = algorithm_name
        self.algorithm_request_dto = algorithm_request_dto

    # The 'start' method is used as a wrapper, so that we can execute the controller
    # while passing the strategy as it's input. That way, we don't need to carry
    # the controller + strategy when we want to execute an algorithm.
    async def start(self):
        await self.controller.exec_algorithm(
            self.algorithm_name, self.algorithm_request_dto, self
        )

    @abstractmethod
    async def run(
        self,
        request_id: str,
        context_id: str,
        algorithm_name: str,
        algorithm_request_dto,
        task_handlers: list,
        metadata,
        logger,
    ) -> str:
        pass
