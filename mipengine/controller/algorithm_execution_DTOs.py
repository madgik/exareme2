from typing import List

from pydantic import BaseModel

from mipengine.controller.api.algorithm_request_dto import AlgorithmRequestDTO
from mipengine.controller.node_tasks_handler_interface import INodeTasksHandler


# One of the two expected data object for the AlgorithmExecutor layer.
class AlgorithmExecutionDTO(BaseModel):
    request_id: str
    context_id: str
    algorithm_name: str
    algorithm_request_dto: AlgorithmRequestDTO

    class Config:
        arbitrary_types_allowed = True


# The second expected data object for the AlgorithmExecutor layer.
# It contains the handler objects(essentially the celery objects) on which the
# AlgorithmExecutor will request tasks execution
class NodesTasksHandlersDTO(BaseModel):
    global_node_tasks_handler: INodeTasksHandler
    local_nodes_tasks_handlers: List[INodeTasksHandler]

    class Config:
        arbitrary_types_allowed = True
