from typing import List

from exareme2 import exaflow_algorithm_classes
from exareme2.controller.federation_info_logs import log_experiment_execution
from exareme2.controller.services.exaflow import ExaflowController
from exareme2.controller.services.exaflow.algorithm_flow_engine_interface import (
    ExaflowAlgorithmFlowEngineInterface,
)
from exareme2.controller.services.exaflow.tasks_handler import ExaflowTasksHandler
from exareme2.controller.services.strategy_interface import AlgorithmExecutionStrategyI


class ExaflowStrategy(AlgorithmExecutionStrategyI):
    _controller: ExaflowController
    _local_worker_tasks_handlers: List[ExaflowTasksHandler]
    _global_worker_tasks_handler: ExaflowTasksHandler

    async def execute(self) -> str:
        variable_names = (self._algorithm_request_dto.inputdata.x or []) + (
            self._algorithm_request_dto.inputdata.y or []
        )
        metadata = self._controller.worker_landscape_aggregator.get_metadata(
            data_model=self._algorithm_request_dto.inputdata.data_model,
            variable_names=variable_names,
        )

        engine = ExaflowAlgorithmFlowEngineInterface(
            request_id=self._request_id,
            context_id=self._context_id,
            tasks_handlers=self._local_worker_tasks_handlers,
        )
        algorithm_cls = exaflow_algorithm_classes[self._algorithm_name]
        algorithm = algorithm_cls(
            inputdata=self._algorithm_request_dto.inputdata,
            engine=engine,
        )
        log_experiment_execution(
            self._logger,
            self._request_id,
            self._context_id,
            self._algorithm_name,
            self._algorithm_request_dto.inputdata.datasets,
            self._algorithm_request_dto.parameters,
            [h.worker_id for h in self._local_worker_tasks_handlers],
        )
        result = algorithm.run(metadata)
        self._logger.info(
            f"Execution completed: {self._algorithm_name} ({self._request_id})"
        )
        return result.json()
