from quart import Quart

from DTOs import AlgorithmDTO
from controller.algorithms import Algorithms

app = Quart(__name__)


@app.route("/algorithms")
async def get_algorithms() -> str:
    algorithm_DTOs = [AlgorithmDTO(algorithm, Algorithms().crossvalidation)
                      for algorithm in Algorithms().available.values()]

    return AlgorithmDTO.schema().dumps(algorithm_DTOs, many=True)
