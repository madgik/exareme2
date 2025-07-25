from typing import Type

from exareme2.algorithms.specifications import AlgorithmType
from exareme2.algorithms.specifications import TransformerName
from exareme2.controller.services.api.algorithm_request_dtos import AlgorithmRequestDTO
from exareme2.controller.services.api.algorithm_spec_dtos import specifications
from exareme2.controller.services.controller_interface import ControllerI
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
from exareme2.controller.services.exareme2.strategies import (
    Exareme2AlgorithmExecutionStrategy,
)
from exareme2.controller.services.exareme2.strategies import LongitudinalStrategy
from exareme2.controller.services.exareme2.strategies import SingleAlgorithmStrategy
from exareme2.controller.services.flower import (
    get_flower_controller as get_flower_controller,
)
from exareme2.controller.services.flower.strategies import FlowerStrategy
from exareme2.controller.services.strategy_interface import AlgorithmExecutionStrategyI
from exareme2.controller.uid_generator import UIDGenerator


def get_algorithm_execution_strategy(
    algorithm_name: str,
    algorithm_request_dto: AlgorithmRequestDTO,
) -> AlgorithmExecutionStrategyI:
    if not algorithm_request_dto.request_id:
        algorithm_request_dto.request_id = UIDGenerator().get_a_uid()

    algo_type = specifications.get_algorithm_type(algorithm_name)
    controller = _get_algorithm_controller(algo_type)
    strategy_type = _get_algorithm_strategy_type(algo_type, algorithm_request_dto)

    return strategy_type(controller, algorithm_name, algorithm_request_dto)


def _get_algorithm_controller(algo_type: AlgorithmType) -> ControllerI:
    controller: ControllerI
    if algo_type in [AlgorithmType.EXAFLOW, AlgorithmType.EXAFLOW_AGGREGATOR]:
        return get_exaflow_controller()
    elif algo_type == AlgorithmType.FLOWER:
        return get_flower_controller()
    elif algo_type == AlgorithmType.EXAREME2:
        return get_exareme2_controller()

    raise NotImplementedError(
        f"Could not get algorithm controller. Unsupported algorithm type: {algo_type}"
    )


def _get_algorithm_strategy_type(
    algo_type: AlgorithmType,
    algorithm_request_dto: AlgorithmRequestDTO,
) -> Type[AlgorithmExecutionStrategyI]:
    strategy: AlgorithmExecutionStrategyI
    if algo_type == AlgorithmType.EXAFLOW:
        return ExaflowStrategy
    elif algo_type == AlgorithmType.EXAFLOW_AGGREGATOR:
        return ExaflowWithAggregationServerStrategy
    elif algo_type == AlgorithmType.FLOWER:
        return FlowerStrategy
    elif algo_type == AlgorithmType.EXAREME2:
        return _get_exareme2_algorithm_strategy_type(algorithm_request_dto)

    raise NotImplementedError(
        f"Could not get algorithm strategy type. Unsupported algorithm type: {algo_type}"
    )


def _get_exareme2_algorithm_strategy_type(
    algorithm_request_dto: AlgorithmRequestDTO,
) -> Type[Exareme2AlgorithmExecutionStrategy]:
    if algorithm_request_dto.preprocessing and algorithm_request_dto.preprocessing.get(
        TransformerName.LONGITUDINAL_TRANSFORMER
    ):
        return LongitudinalStrategy
    else:
        return SingleAlgorithmStrategy
