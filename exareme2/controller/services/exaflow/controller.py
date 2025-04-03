from typing import Dict
from typing import List

from exareme2 import exaflow_algorithm_classes
from exareme2.controller import logger as ctrl_logger
from exareme2.controller.federation_info_logs import log_experiment_execution
from exareme2.controller.services.exaflow.execution_engine import (
    AlgorithmExecutionEngine,
)
from exareme2.controller.services.exaflow.tasks_handler import TasksHandler
from exareme2.controller.uid_generator import UIDGenerator
from exareme2.worker_communication import WorkerInfo


class Controller:
    def __init__(self, worker_landscape_aggregator, task_timeout):
        self.worker_landscape_aggregator = worker_landscape_aggregator
        self.task_timeout = task_timeout

    def _create_worker_tasks_handler(self, request_id, worker_info: WorkerInfo):
        worker_addr = f"{worker_info.ip}:{worker_info.port}"
        return TasksHandler(
            request_id,
            worker_id=worker_info.id,
            worker_queue_addr=worker_addr,
            tasks_timeout=self.task_timeout,
        )

    async def exec_algorithm(self, algorithm_name, algorithm_request_dto):
        request_id = algorithm_request_dto.request_id
        context_id = UIDGenerator().get_a_uid()
        logger = ctrl_logger.get_request_logger(request_id)

        datasets = algorithm_request_dto.inputdata.datasets

        csv_paths_per_worker_id: Dict[
            str, List[str]
        ] = self.worker_landscape_aggregator.get_csv_paths_per_worker_id(
            algorithm_request_dto.inputdata.data_model, datasets
        )

        workers_info = [
            self.worker_landscape_aggregator.get_worker_info(worker_id)
            for worker_id in csv_paths_per_worker_id
        ]

        task_handlers = [
            self._create_worker_tasks_handler(request_id, worker)
            for worker in workers_info
        ]

        engine = AlgorithmExecutionEngine(
            request_id=request_id,
            context_id=context_id,
            tasks_handlers=task_handlers,
            csv_paths_per_worker_id=csv_paths_per_worker_id,
        )

        algorithm_class = exaflow_algorithm_classes[algorithm_name]
        algorithm = algorithm_class(
            inputdata=algorithm_request_dto.inputdata,
            engine=engine,
        )

        variable_names = (algorithm_request_dto.inputdata.x or []) + (
            algorithm_request_dto.inputdata.y or []
        )
        metadata = self.worker_landscape_aggregator.get_metadata(
            data_model=algorithm_request_dto.inputdata.data_model,
            variable_names=variable_names,
        )
        try:
            log_experiment_execution(
                logger,
                request_id,
                context_id,
                algorithm_name,
                algorithm_request_dto.inputdata.datasets,
                algorithm_request_dto.parameters,
                [info.id for info in workers_info],
            )

            result = algorithm.run(
                inputdata=algorithm_request_dto.inputdata.dict(),
                metadata=metadata,
            )

            logger.info(f"Finished execution -> {algorithm_name} with {request_id}")
            return result.json()

        except Exception as e:
            logger.exception(f"Algorithm execution failed: {str(e)}")
            raise
