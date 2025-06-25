from typing import List

from exareme2 import exaflow_algorithm_classes
from exareme2.aggregation_clients.controller_aggregation_client import (
    ControllerAggregationClient,
)
from exareme2.controller.federation_info_logs import log_experiment_execution
from exareme2.controller.services.exaflow import ExaflowController
from exareme2.controller.services.exaflow.execution_engine import (
    ExaflowAlgorithmFlowEngineInterface,
)
from exareme2.controller.services.exaflow.tasks_handler import ExaflowTasksHandler
from exareme2.controller.services.strategy_interface import AlgorithmExecutionStrategyI


class ExaflowStrategy(AlgorithmExecutionStrategyI):
    controller: ExaflowController
    tasks_handlers: List[ExaflowTasksHandler]

    async def execute(self) -> str:
        variable_names = (self.algorithm_request_dto.inputdata.x or []) + (
            self.algorithm_request_dto.inputdata.y or []
        )
        metadata = self.controller.worker_landscape_aggregator.get_metadata(
            data_model=self.algorithm_request_dto.inputdata.data_model,
            variable_names=variable_names,
        )

        engine = ExaflowAlgorithmFlowEngineInterface(
            request_id=self.request_id,
            context_id=self.context_id,
            tasks_handlers=self.tasks_handlers,
        )
        algorithm_cls = exaflow_algorithm_classes[self.algorithm_name]
        algorithm = algorithm_cls(
            inputdata=self.algorithm_request_dto.inputdata,
            engine=engine,
        )
        log_experiment_execution(
            self.logger,
            self.request_id,
            self.context_id,
            self.algorithm_name,
            self.algorithm_request_dto.inputdata.datasets,
            self.algorithm_request_dto.parameters,
            [h.worker_id for h in self.tasks_handlers],
        )
        result = algorithm.execute(metadata)
        self.logger.info(
            f"Execution completed: {self.algorithm_name} ({self.request_id})"
        )
        return result.json()


class ExaflowWithAggregationServerStrategy(ExaflowStrategy):
    async def execute(self) -> str:
        agg_client = ControllerAggregationClient(self.request_id)
        status = agg_client.configure(num_workers=len(self.tasks_handlers))
        if status != "Configured":
            raise RuntimeError(f"AggregationServer refused to configure: {status}")
        self.logger.debug(f"Aggregation configured: {status}")

        try:
            return await super().execute()
        finally:
            cleanup_status = agg_client.cleanup()
            self.logger.debug(f"Aggregation cleanup response: {cleanup_status}")
