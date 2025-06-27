from abc import ABC
from abc import abstractmethod
from logging import Logger
from typing import List
from typing import Optional

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

    _controller: ControllerI
    _algorithm_name: str
    _algorithm_request_dto: AlgorithmRequestDTO
    _request_id: str
    _context_id: str
    _logger: Logger
    _local_worker_tasks_handlers: List[TasksHandlerI]
    _global_worker_tasks_handler: Optional[TasksHandlerI]

    def __init__(
        self,
        controller: ControllerI,
        algorithm_name: str,
        algorithm_request_dto: AlgorithmRequestDTO,
    ):
        self._controller = controller
        self._algorithm_name = algorithm_name
        self._algorithm_request_dto = algorithm_request_dto
        self._request_id = self._algorithm_request_dto.request_id
        self._context_id = UIDGenerator().get_a_uid()
        self._logger = ctrl_logger.get_request_logger(self._request_id)
        self._local_worker_tasks_handlers = (
            self._controller.get_local_worker_tasks_handlers(
                self._algorithm_request_dto.inputdata.data_model,
                self._algorithm_request_dto.inputdata.datasets,
                self._request_id,
            )
        )
        self._global_worker_tasks_handler = (
            self._controller.get_global_worker_tasks_handler(self._request_id)
        )

    @abstractmethod
    async def execute(
        self,
    ) -> str:
        pass
