from typing import List
from typing import Type

from exaflow.algorithms.specifications import AlgorithmType
from exaflow.algorithms.specifications import ComponentType
from exaflow.algorithms.specifications import TransformerName
from exaflow.controller.services.api.algorithm_request_dtos import AlgorithmRequestDTO
from exaflow.controller.services.api.algorithm_spec_dtos import specifications
from exaflow.controller.services.controller_interface import ControllerI
from exaflow.controller.services.exareme3 import (
    get_exaflow_controller as get_exaflow_controller,
)
from exaflow.controller.services.exareme3.strategies import ExaflowStrategy
from exaflow.controller.services.exareme3.strategies import (
    ExaflowWithAggregationServerStrategy,
)
from exaflow.controller.services.flower import (
    get_flower_controller as get_flower_controller,
)
from exaflow.controller.services.flower.strategies import FlowerStrategy
from exaflow.controller.services.strategy_interface import AlgorithmExecutionStrategyI
from exaflow.controller.uid_generator import UIDGenerator


def get_algorithm_execution_strategy(
    algorithm_name: str,
    algorithm_request_dto: AlgorithmRequestDTO,
) -> AlgorithmExecutionStrategyI:
    if not algorithm_request_dto.request_id:
        algorithm_request_dto.request_id = UIDGenerator().get_a_uid()

    algo_type = specifications.get_algorithm_type(algorithm_name)
    components = specifications.get_component_types(algorithm_name)
    controller = _get_algorithm_controller(algo_type)
    strategy_type = _get_algorithm_strategy_type(
        algo_type, components, algorithm_request_dto
    )

    return strategy_type(controller, algorithm_name, algorithm_request_dto)


def _get_algorithm_controller(algo_type: AlgorithmType) -> ControllerI:
    controller: ControllerI
    if algo_type in [AlgorithmType.EXAFLOW]:
        return get_exaflow_controller()
    elif algo_type == AlgorithmType.FLOWER:
        return get_flower_controller()

    raise NotImplementedError(
        f"Could not get algorithm controller. Unsupported algorithm type: {algo_type}"
    )


def _get_algorithm_strategy_type(
    algo_type: AlgorithmType,
    algo_component_types: List[ComponentType],
    algorithm_request_dto: AlgorithmRequestDTO,
) -> Type[AlgorithmExecutionStrategyI]:
    strategy: AlgorithmExecutionStrategyI
    if algo_type == AlgorithmType.EXAFLOW:
        if ComponentType.AGGREGATION_SERVER in algo_component_types:
            return ExaflowWithAggregationServerStrategy
        return ExaflowStrategy
    elif algo_type == AlgorithmType.FLOWER:
        return FlowerStrategy

    raise NotImplementedError(
        f"Could not get algorithm strategy type. Unsupported algorithm type: {algo_type}"
    )
