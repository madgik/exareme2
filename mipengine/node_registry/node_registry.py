import asyncio
import copy

from mipengine.common.node_registry_DTOs import NodeRecord, NodeRecordsList


class NodeRegistry:
    # singleton
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(NodeRegistry, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self._registry = NodeRecordsList(node_records=[])
        self._registry_lock = asyncio.Lock()

    async def register_node(self, node_record: NodeRecord):
        async with self._registry_lock:
            for nrecord in self._registry.node_records:
                if nrecord.node_id == node_record.node_id:
                    raise self.NodeIdAlreadyInRegistryError(nrecord)
                elif nrecord.task_queue_ip == node_record.task_queue_ip:
                    raise self.TaskQueueIPAlreadyInUseError(nrecord)
                elif nrecord.db_ip == node_record.db_ip:
                    raise self.DatabaseIPAlreadyInUseError(nrecord)

            self._registry.node_records.append(node_record)

    async def deregister_node(self, node_id):
        self._registry = NodeRecordsList(
            node_records=[
                node_record
                for node_record in self._registry.node_records
                if node_record.node_id != node_id
            ]
        )

    # async def update_node(self, node_id: str, node_record: NodeRecord):
    #     if node_id not in self._registry:
    #         raise self.UpdateUnknownNodeError()
    #     else:
    #         self._registry[node_id] = node_record

    # async def get_node(self, node_id):
    #     try:
    #         return copy.deepcopy(self._registry[node_id])
    #     except KeyError:
    #         raise self.NodeIdNotInResgistry

    async def get_all_nodes(self):
        return copy.deepcopy(self._registry)

    class TaskQueueIPAlreadyInUseError(Exception):
        def __init__(self, node_record):
            self.message = (
                f"Node with node_id:{node_record.node_id} is already registered with "
                "task_queue_ip: {node_record.task_queue_ip}"
            )

    class DatabaseIPAlreadyInUseError(Exception):
        def __init__(self, node_record):
            self.message = (
                f"Node with node_id:{node_record.node_id} is already registered with "
                "db_ip: {node_record.db_ip}"
            )

    class NodeIdAlreadyInRegistryError(Exception):
        def __init__(self, node_record):
            self.message = (
                f"A node with id:{node_record.node_id} already exists in the "
                "NodeRegistry. Either first deregister the node and then register it "
                "or update enregistered node"
            )

    class NodeIdNotInResgistry(Exception):
        def __init__(self, node_record):
            self.message = (
                f"NodeId: {node_record.node_id} does not exist in the node registry"
            )

    class UpdateUnknownNodeIdError(Exception):
        def __init__(self, node_record):
            self.message = (
                f"There are no registered nodes with id: {node_record.node_id}"
            )
