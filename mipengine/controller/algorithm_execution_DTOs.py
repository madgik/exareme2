from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel

from mipengine.algorithms.algorithm import Variables
from mipengine.controller.algorithm_execution_tasks_handler import (
    INodeAlgorithmTasksHandler,
)

# One of the two expected data object for the AlgorithmExecutor layer.
# class AlgorithmExecutionDTO(BaseModel):
#     request_id: str
#     context_id: str
#     algorithm_name: str
#     data_model: str
#     # datasets_per_local_node: Dict[str, List[str]]
#     variables:Variables
#     var_filters: Optional[dict] = None
#     algo_parameters: Optional[Dict[str, Any]] = None
#     algo_flags: Optional[Dict[str, Any]] = None
#     metadata:Dict[str,dict]

#     class Config:
#         arbitrary_types_allowed = True


# The second expected data object for the AlgorithmExecutor layer.
# It contains the handler objects(essentially the celery objects) on which the
# AlgorithmExecutor will request tasks execution
class NodesTasksHandlersDTO(BaseModel):
    global_node_tasks_handler: Optional[INodeAlgorithmTasksHandler]
    local_nodes_tasks_handlers: List[INodeAlgorithmTasksHandler]

    class Config:
        arbitrary_types_allowed = True
