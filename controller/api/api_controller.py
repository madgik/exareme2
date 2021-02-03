from quart import Quart, request

from controller.algorithms import Algorithms
from controller.api.DTOs.AlgorithmDTO import AlgorithmDTO
from controller.api.DTOs.AlgorithmExecutionDTOs import AlgorithmRequestDTO
from controller.api.errors import BadRequest

app = Quart(__name__)


# TODO break into views/app/errors


@app.route("/algorithms")
async def get_algorithms() -> str:
    algorithm_DTOs = [AlgorithmDTO(algorithm, Algorithms().crossvalidation)
                      for algorithm in Algorithms().available.values()]

    return AlgorithmDTO.schema().dumps(algorithm_DTOs, many=True)


@app.route("/algorithms/<algorithm_name>", methods=['POST'])
async def run_algorithm(algorithm_name: str) -> str:
    algorithm_request = AlgorithmRequestDTO.from_json(await request.data)
    print(f"Running algorithm: {algorithm_name} with body: {algorithm_request}")

    if str.lower(algorithm_name) not in Algorithms().available.keys():
        raise BadRequest(f"Algorithm '{algorithm_name}' does not exist.")
    response = run_algorithm(algorithm_request)
    return "Response"


@app.errorhandler(BadRequest)
def handle_bad_request(error: BadRequest):
    return error.message, error.status_code
