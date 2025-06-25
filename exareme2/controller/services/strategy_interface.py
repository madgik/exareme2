from abc import ABC
from abc import abstractmethod
from logging import Logger
from typing import List
from typing import Optional

from billiard.pool import TaskHandler

from exareme2.controller import logger as ctrl_logger
from exareme2.controller.services.api.algorithm_request_dtos import AlgorithmRequestDTO
from exareme2.controller.services.controller_interface import ControllerI
from exareme2.controller.services.tasks_handler_interface import TasksHandlerI
from exareme2.controller.uid_generator import UIDGenerator


class AlgorithmExecutionStrategyI(ABC):
    """
    The AlgorithmExecutionStrategy holds algorithm execution specific information. It is created and deleted
    along with the algorithm execution lifecycle.
    The Controller class is passed in the strategy init method, and it is used to allow the strategy to use some
    algorithm execution independent variables.
    """

    controller: ControllerI
    algorithm_name: str
    algorithm_request_dto: AlgorithmRequestDTO
    request_id: str
    context_id: str
    logger: Logger
    local_worker_tasks_handlers: List[TasksHandlerI]
    global_worker_tasks_handler: Optional[TasksHandlerI]

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
        self.local_worker_tasks_handlers = (
            self.controller.get_local_worker_tasks_handlers(
                self.algorithm_request_dto.inputdata.data_model,
                self.algorithm_request_dto.inputdata.datasets,
                self.request_id,
            )
        )
        self.global_worker_tasks_handler = (
            self.controller.get_global_worker_tasks_handler(self.request_id)
        )

    @abstractmethod
    async def execute(
        self,
    ) -> str:
        pass
