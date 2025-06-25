from typing import Optional

from exareme2.algorithms.specifications import AlgorithmType
from exareme2.controller import config as ctrl_config
from exareme2.controller.services import get_worker_landscape_aggregator
from exareme2.controller.services.api.algorithm_request_dtos import AlgorithmRequestDTO
from exareme2.controller.services.api.algorithm_request_validator import (
    validate_algorithm_request,
)
from exareme2.controller.services.api.algorithm_spec_dtos import specifications
from exareme2.controller.services.controller_interface import ControllerI
from exareme2.controller.services.exaflow import (
    get_aggregation_server_exaflow_controller,
)
from exareme2.controller.services.exaflow import (
    get_exaflow_controller as get_exaflow_controller,
)
from exareme2.controller.services.exaflow.strategies import ExaflowStrategy
from exareme2.controller.services.exaflow.strategies import (
    ExaflowWithAggregationServerStrategy,
)
from exareme2.controller.services.exareme2 import (
    get_controller as get_exareme2_controller,
)
from exareme2.controller.services.flower import get_controller as get_flower_controller
from exareme2.controller.services.strategy_interface import AlgorithmExecutionStrategyI
from exareme2.controller.uid_generator import UIDGenerator


def get_algorithm_execution_strategy(
    algo_name: str, request_dto: AlgorithmRequestDTO
) -> AlgorithmExecutionStrategyI:
    if not request_dto.request_id:
        request_dto.request_id = UIDGenerator().get_a_uid()

    validate_algorithm_request(
        algorithm_name=algo_name,
        algorithm_request_dto=request_dto,
        algorithms_specs=specifications.enabled_algorithms,
        transformers_specs=specifications.enabled_transformers,
        worker_landscape_aggregator=get_worker_landscape_aggregator(),
        smpc_enabled=ctrl_config.smpc.enabled,
        smpc_optional=ctrl_config.smpc.optional,
    )

    algo_type = specifications.get_algorithm_type(algo_name)
    controller = _get_algorithm_controller(algo_type)
    strategy_type = _get_algorithm_strategy(algo_type, controller)

    if strategy_type is None:
        return None

    return strategy


def _get_algorithm_controller(algo_type: AlgorithmType) -> ControllerI:
    controller: ControllerI
    if algo_type in [AlgorithmType.EXAFLOW, AlgorithmType.EXAFLOW_AGGREGATOR]:
        return get_exaflow_controller()
    elif algo_type == AlgorithmType.FLOWER:
        return get_flower_controller()
    elif algo_type == AlgorithmType.EXAREME2:
        return get_exareme2_controller()

    raise NotImplementedError(f"Unsupported algorithm type: {algo_type}")


def _get_algorithm_strategy(
    algo_type: AlgorithmType, controller: ControllerI
) -> Optional[Type[AlgorithmExecutionStrategyI]]:
    strategy: AlgorithmExecutionStrategyI
    if algo_type == AlgorithmType.EXAFLOW:
        return ExaflowStrategy
    elif algo_type == AlgorithmType.EXAFLOW_AGGREGATOR:
        return ExaflowWithAggregationServerStrategy
    elif algo_type == AlgorithmType.FLOWER:
        return None
    elif algo_type == AlgorithmType.EXAREME2:
        return get_exareme2_controller()

    raise NotImplementedError(f"Unsupported algorithm type: {algo_type}")
