from exareme2.algorithms.specifications import AlgorithmType
from exareme2.controller import config as ctrl_config
from exareme2.controller.services import get_worker_landscape_aggregator
from exareme2.controller.services.api.algorithm_request_dtos import AlgorithmRequestDTO
from exareme2.controller.services.api.algorithm_request_validator import (
    validate_algorithm_request,
)
from exareme2.controller.services.api.algorithm_spec_dtos import specifications
from exareme2.controller.services.exaflow import (
    get_controller as get_exaflow_controller,
)
from exareme2.controller.services.exareme2 import (
    get_controller as get_exareme2_controller,
)
from exareme2.controller.services.flower import get_controller as get_flower_controller
from exareme2.controller.uid_generator import UIDGenerator


async def execute_algorithm(algo_name: str, request_dto: AlgorithmRequestDTO):
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

    if request_dto.type == AlgorithmType.FLOWER:
        controller = get_flower_controller()
    elif request_dto.type == AlgorithmType.EXAFLOW:
        controller = get_exaflow_controller()
    else:
        controller = get_exareme2_controller()

    algorithm_result = await controller.exec_algorithm(
        algorithm_name=algo_name,
        algorithm_request_dto=request_dto,
    )

    return algorithm_result
