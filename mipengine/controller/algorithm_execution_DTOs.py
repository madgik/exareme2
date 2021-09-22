from ipaddress import IPv4Address
from pydantic import BaseModel
from typing import Dict, List, Tuple, Optional, Any

from mipengine.controller.node_tasks_handler_interface import INodeTasksHandler

# AlgorithmRequestDTO is the expected data format for the
# webaAPI(algorithms_endpoints.py::post_algoithm) layer. The webAPI propagates
# this dto to the Controller layer.
class AlgorithmRequestDTO(BaseModel):
    pathology: str
    datasets: List[str]
    x: List[str]
    y: List[str]
    filters: dict = None  # TODO this should be better(more strictly) defined
    algorithm_params: dict = None  # TODO this should be better(more strictly) defined


# AlgorithmExecutionDTO is one of the two expected data object for the
# AlgorithmExecutor layer.
class AlgorithmExecutionDTO(BaseModel):
    context_id: str
    algorithm_name: str
    algorithm_request_dto: AlgorithmRequestDTO

    class Config:
        arbitrary_types_allowed = True


# NodesTasksHandlersDTO is the second expected data object for the AlgorithmExecutor
# layer. It contains the handler objects(essentially the celery objects) on which the
# AlgorithmExecutor can request tasks execution
class NodesTasksHandlersDTO(BaseModel):
    global_node_tasks_handler: INodeTasksHandler
    local_nodes_tasks_handlers: List[INodeTasksHandler]

    class Config:
        arbitrary_types_allowed = True
