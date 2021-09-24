from pydantic import BaseModel
from typing import Dict, List


# AlgorithmRequestDTO is the expected data format for the
# webaAPI(algorithms_endpoints.py::post_algoithm) layer. The webAPI propagates
# this dto to the Controller layer.
class AlgorithmRequestDTO(BaseModel):
    pathology: str
    datasets: List[str]
    x: List[str]
    y: List[str]
    filters: dict = None  # TODO Could this be better(more strictly) defined
    algorithm_params: Dict[str, List[str]] = None
