from abc import ABC
from abc import abstractmethod
from logging import Logger
from typing import List

from billiard.pool import TaskHandler

from exareme2.controller import logger as ctrl_logger
from exareme2.controller.services.api.algorithm_request_dtos import AlgorithmRequestDTO
from exareme2.controller.services.controller_interface import ControllerI
from exareme2.controller.uid_generator import UIDGenerator


# The AlgorithmExecutionStrategy holds algorithm execution specific information. It is created and deleted
# along with the algorithm execution lifecycle.
# The Controller class is passed in the strategy init method, and it is used to connect the strategy with the workers.
class AlgorithmExecutionStrategyI(ABC):
    controller: ControllerI
    algorithm_name: str
    algorithm_request_dto: AlgorithmRequestDTO
    request_id: str
    context_id: str
    logger: Logger
    tasks_handlers: List[TaskHandler]

    def __init__(
        self,
        controller: ControllerI,
        algorithm_name: str,
        algorithm_request_dto: AlgorithmRequestDTO,
    ):
        self.controller = controller
        self.algorithm_name = algorithm_name
        self.algorithm_request_dto = algorithm_request_dto
        self.request_id = self.algorithm_request_dto.request_id
        self.context_id = UIDGenerator().get_a_uid()
        self.logger = ctrl_logger.get_request_logger(self.request_id)
        self.tasks_handlers = self.controller.get_tasks_handlers(
            self.algorithm_request_dto.inputdata.data_model,
            self.algorithm_request_dto.inputdata.datasets,
            self.request_id,
        )

    @abstractmethod
    async def execute(
        self,
    ) -> str:
        pass
