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
        # Pathologies are also stored in the consul key/value store
        node_configuration = self._NodeParameters(
            db_id=node_record.db_id,
            pathologies=node_record.pathologies,
        )
        self._consul_kv_store.put(node_record.node_id, node_configuration.json())

    def deregister_node(self, node_id: str):
        _, data = self._consul_kv_store.get(node_id)

        if not data:
            raise self.NodeIDNotInKVStore(node_id)

        node_conf = self._NodeParameters.parse_raw(data["Value"])

        # deregister db service
        self._consul_service.deregister(node_conf.db_id)

        # deregister node service
        self._consul_service.deregister(service_id=node_id)

        # delete node configuration from kv store
        self._consul_kv_store.delete(node_id)

    def get_all_nodes(self) -> Dict[str, "NodeInfo"]:
        all_services = self._consul_agent.services()

        all_nodes = self._parse_nodes_from_services(all_services)

        # pathologies are read from the key value store
        for node_id in all_nodes.keys():
            _, data = self._consul_kv_store.get(node_id)
            if not data:
                raise self.NodeIDNotInKVStore(node_id)
            node_params = self._NodeParameters.parse_raw(data["Value"])
            all_nodes[node_id].pathologies = node_params.pathologies

        return all_nodes

    def _parse_nodes_from_services(
        self, all_services: Dict[str, str]
    ) -> Dict[str, "NodeInfo"]:
        nodes = {}
        for (service_id, service_info) in all_services.items():
            tags = service_info["Tags"]
            role = ""
            if NodeRole.GLOBALNODE.name in tags:
                role = NodeRole.GLOBALNODE
            elif NodeRole.LOCALNODE.name in tags:
                role = NodeRole.LOCALNODE
            else:
                continue

            node_info = self.NodeInfo(
                ip=service_info["Address"],
                port=service_info["Port"],
                role=role,
            )
            nodes[service_id] = node_info

        return nodes

    def get_all_global_nodes(self):
        all_nodes = self.get_all_nodes()
        return {
            node_id: node_info
            for node_id, node_info in all_nodes.items()
            if node_info.role == NodeRole.GLOBALNODE
        }

    def get_all_local_nodes(self):
        all_nodes = self.get_all_nodes()
        return {
            node_id: node_info
            for node_id, node_info in all_nodes.items()
            if node_info.role == NodeRole.LOCALNODE
        }

    def get_nodes_with_pathologies(self, pathologies: List[str]):
        all_local_nodes = self.get_all_local_nodes()

        local_nodes_with_pathologies = {}
        for node_id, node_info in all_local_nodes.items():
            if all(p in [p.name for p in node_info.pathologies] for p in pathologies):
                local_nodes_with_pathologies[node_id] = node_info

        return local_nodes_with_pathologies

    def get_nodes_with_datasets(self, datasets: List[str]):
        all_local_nodes = self.get_all_local_nodes()

        local_nodes_with_datasets = {}
        for node_id, node_info in all_local_nodes.items():
            node_datasets = []
            for p in node_info.pathologies:
                node_datasets.extend(p.datasets)
            if all(d in node_datasets for d in datasets):
                local_nodes_with_datasets[node_id] = node_info
        return local_nodes_with_datasets

    def get_db(self, node_id: str) -> "DBInfo":
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

    def get_dbs(self, node_ids: List[str]) -> Dict[str, "DBInfo"]:
        return {node_id: self.get_db(node_id) for node_id in node_ids}

    class NodeIDNotInKVStore(Exception):
        def __init__(self, node_id: str):
            self.message = f"There is no node_id:{node_id} key in the key/value store"

    class NodeInfo(BaseModel):
        ip: IPv4Address
        port: int
        role: NodeRole
        pathologies: Optional[List[Pathology]]

    class _NodeParameters(BaseModel):
        db_id: str
        pathologies: Optional[List[Pathology]]

    class DBInfo(BaseModel):
        id: str
        ip: IPv4Address
        port: int
