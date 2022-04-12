from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel

from mipengine.controller.node_tasks_handler_interface import INodeTasksHandler


# One of the two expected data object for the AlgorithmExecutor layer.
class AlgorithmExecutionDTO(BaseModel):
    request_id: str
    context_id: str
    algorithm_name: str
    data_model: str
    datasets_per_local_node: Dict[str, List[str]]
    x_vars: Optional[List[str]] = None
    y_vars: Optional[List[str]] = None
    var_filters: dict = None
    algo_parameters: Optional[Dict[str, Any]] = None
    algo_flags: Optional[Dict[str, Any]] = None

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
