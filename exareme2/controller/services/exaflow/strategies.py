from exareme2 import exaflow_algorithm_classes
from exareme2.aggregation_clients.controller_aggregation_client import (
    ControllerAggregationClient,
)
from exareme2.controller.federation_info_logs import log_experiment_execution
from exareme2.controller.services.exaflow.execution_engine import (
    AlgorithmExecutionEngine,
)
from exareme2.controller.services.strategy_interface import AlgorithmExecutionStrategyI


class ExaflowStrategy(AlgorithmExecutionStrategyI):
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
        engine = AlgorithmExecutionEngine(
            request_id=request_id,
            context_id=context_id,
            tasks_handlers=task_handlers,
        )
        algorithm_cls = exaflow_algorithm_classes[algorithm_name]
        algorithm = algorithm_cls(
            inputdata=algorithm_request_dto.inputdata,
            engine=engine,
        )
        log_experiment_execution(
            logger,
            request_id,
            context_id,
            algorithm_name,
            algorithm_request_dto.inputdata.datasets,
            algorithm_request_dto.parameters,
            [h.worker_id for h in task_handlers],
        )
        result = algorithm.run(metadata)
        logger.info(f"Execution completed: {algorithm_name} ({request_id})")
        return result.json()


class ExaflowWithAggregationServerStrategy(ExaflowStrategy):
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
        agg_client = ControllerAggregationClient(request_id)
        status = agg_client.configure(num_workers=len(task_handlers))
        if status != "Configured":
            raise RuntimeError(f"AggregationServer refused to configure: {status}")
        logger.debug(f"Aggregation configured: {status}")

        try:
            return await super().run(
                request_id,
                context_id,
                algorithm_name,
                algorithm_request_dto,
                task_handlers,
                metadata,
                logger,
            )
        finally:
            cleanup_status = agg_client.cleanup()
            logger.debug(f"Aggregation cleanup response: {cleanup_status}")
