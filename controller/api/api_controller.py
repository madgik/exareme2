import json
from dataclasses import asdict

from quart import Quart

from DTOs import AlgorithmDTO
from controller.algorithms import Algorithms

app = Quart(__name__)


@app.route("/algorithms")
async def get_algorithms() -> str:
    algorithm_DTOs = [asdict(AlgorithmDTO(algorithm, Algorithms().crossvalidation))
                      for algorithm in Algorithms().available.values()]

    # return algorithm_DTOs[0].to_json()
    return json.dumps(algorithm_DTOs)
    # return AlgorithmDTO.schema().dumps(algorithm_DTOs, many=True)
