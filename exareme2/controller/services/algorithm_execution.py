from exareme2.controller import config as ctrl_config
from exareme2.controller.services import get_worker_landscape_aggregator
from exareme2.controller.services.api.algorithm_request_dtos import AlgorithmRequestDTO
from exareme2.controller.services.api.algorithm_request_validator import (
    validate_algorithm_request,
)
from exareme2.controller.services.api.algorithm_spec_dtos import specifications
from exareme2.controller.services.exareme2 import (
    get_controller as get_exareme2_controller,
)
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
    algorithm_result = await get_exareme2_controller().exec_algorithm(
        algorithm_name=algo_name,
        algorithm_request_dto=request_dto,
    )

    return algorithm_result
