import consul
from ipaddress import IPv4Address
from typing import Optional, List, Dict
from pydantic import BaseModel

from mipengine.common.node_registry_DTOs import (
    NodeRecord,
    Pathology,
    NodeRole,
)


class NodeRegistryClient:
    def __init__(
        self,
        consul_server_ip: IPv4Address = IPv4Address("127.0.0.1"),
        consul_server_port: int = 8500,
    ):
        c = consul.Consul(host=consul_server_ip, port=consul_server_port)
        self._consul_agent = c.agent
        self._consul_service = c.agent.service
        self._consul_kv_store = c.kv

    def register_node(self, node_record: NodeRecord):
        # register the node as a service
        self._consul_service.register(
            name=node_record.node_id,
            service_id=node_record.node_id,
            address=str(node_record.task_queue_ip),
            port=node_record.task_queue_port,
            tags=(node_record.node_role.name,),
        )
        # register the db as a service
        self._consul_service.register(
            name=node_record.db_id,
            service_id=node_record.db_id,
            address=str(node_record.db_ip),
            port=node_record.db_port,
            tags=("db",),
        )

        # The node's db is linked to the node by storing the db_id as a value in the key
        # value store of consul
        # The node's pathologies are also stored in the consul key/value store
        node_configuration = self._NodeParameters(
            db_id=node_record.db_id, pathologies=node_record.pathologies
        )
        self._consul_kv_store.put(node_record.node_id, node_configuration.json())

    def deregister_node(self, node_id: str):
        _, data = self._consul_kv_store.get(node_id)

        if not data:
            raise NodeIDNotInKVStore(node_id)

        node_conf = self._NodeParameters.parse_raw(data["Value"])

        # deregister db service
        self._consul_service.deregister(node_conf.db_id)

        # deregister node service
        self._consul_service.deregister(service_id=node_id)

        # delete node configuration from kv store
        self._consul_kv_store.delete(node_id)

    def get_all_nodes_info(self) -> Dict[str, "NodeInfo"]:
        all_services = self._consul_agent.services()
        node_roles_str = [node_role.name for node_role in list(NodeRole)]
        all_nodes = {
            service_id: self.NodeInfo(
                ip=service_info["Address"], port=service_info["Port"]
            )
            for (service_id, service_info) in all_services.items()
            if any(x in service_info["Tags"] for x in node_roles_str)
        }

        # pathologies are read from the key value store
        for node_id in all_nodes.keys():
            _, data = self._consul_kv_store.get(node_id)
            if not data:
                raise self.NodeIDNotInKVStore(node_id)
            node_conf = self._NodeParameters.parse_raw(data["Value"])
            all_nodes[node_id].pathologies = node_conf.pathologies

        return all_nodes

    def get_db_info(self, node_id: str) -> "DBInfo":
        _, data = self._consul_kv_store.get(node_id, index=None)

        if not data:
            raise self.NodeIDNotInKVStore(node_id)

        node_conf = self._NodeParameters.parse_raw(data["Value"])
        db_id = node_conf.db_id

        all_services = self._consul_agent.services()
        db_service = all_services[db_id]

        db_info = self.DBInfo(
            id=db_service["ID"], ip=db_service["Address"], port=db_service["Port"]
        )
        return db_info

    class NodeIDNotInKVStore(Exception):
        def __init__(self, node_id: str):
            self.message = f"There is no node_id:{node_id} key in the key/value store"

    class NodeInfo(BaseModel):
        ip: IPv4Address
        port: int
        pathologies: Optional[List[Pathology]]

    class _NodeParameters(BaseModel):
        db_id: str
        pathologies: Optional[List[Pathology]]

    class DBInfo(BaseModel):
        id: str
        ip: IPv4Address
        port: int
