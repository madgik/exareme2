import asyncio
import concurrent.futures
import logging

from quart import Blueprint
from quart import request

from mipengine.controller.algorithm_executor.algorithm_executor import AlgorithmExecutor
from mipengine.controller.api.algorithm_request_dto import AlgorithmRequestDTO
from mipengine.controller.api.algorithm_specifications_dtos import (
    AlgorithmSpecificationDTO,
)
from mipengine.controller.api.algorithm_specifications_dtos import (
    algorithm_specificationsDTOs,
)
from mipengine.controller.api.validator import validate_algorithm_request
from mipengine.controller.node_registry import node_registry

algorithms = Blueprint("algorithms_endpoint", __name__)


@algorithms.before_app_serving
async def startup():
    asyncio.create_task(node_registry.update())


@algorithms.route("/datasets")
async def get_datasets() -> dict:
    datasets = {}
    for node in node_registry.get_all_local_nodes():
        datasets[node.id] = node.datasets_per_schema

    return datasets


@algorithms.route("/algorithms")
async def get_algorithms() -> str:
    algorithm_specifications = algorithm_specificationsDTOs.algorithms_list

    return AlgorithmSpecificationDTO.schema().dumps(algorithm_specifications, many=True)


@algorithms.route("/algorithms/<algorithm_name>", methods=["POST"])
async def post_algorithm(algorithm_name: str) -> str:
    logging.info(f"Algorithm execution with name {algorithm_name}.")

    request_body = await request.data

    validate_algorithm_request(algorithm_name, request_body)

    algorithm_request = AlgorithmRequestDTO.from_json(request_body)

    # TODO: This looks freakin awful...
    # Function run_algorithm_executor_in_threadpool calls the run method on the AlgorithmExecutor on a separate thread
    # This function is queued in the running event loop
    # Thus AlgorithmExecutor.run is executed asynchronoysly and does not block further requests to the server
    def run_algorithm_executor_in_threadpool(algorithm_name, algorithm_request):
        alg_ex = AlgorithmExecutor(algorithm_name, algorithm_request)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(alg_ex.run)
            result = future.result()
            return result

    loop = asyncio.get_running_loop()
    algorithm_result = await loop.run_in_executor(
        None,
        run_algorithm_executor_in_threadpool,
        algorithm_name,
        algorithm_request,
    )
    return algorithm_result.json()
