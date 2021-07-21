import consul
from ipaddress import IPv4Address
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from enum import Enum, unique, auto


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

    def register_node(self, node_params: "NodeParams", db_params: "DBParams"):
        # TODO: As is, if a node calls register_node with the same node_params.id as an
        # existing node service, it will overwrite the existing node service parameters

        self._check_params(node_params, db_params)

        # register the node as a service
        self._consul_service.register(
            name=node_params.id,
            service_id=node_params.id,
            address=str(node_params.ip),
            port=node_params.port,
            tags=(node_params.role,),
            enable_tag_override=True,
        )

        # register the db as a service
        self._consul_service.register(
            name=db_params.id,
            service_id=db_params.id,
            address=str(db_params.ip),
            port=db_params.port,
            tags=("db",),
            enable_tag_override=True,
        )

        # The mapping between node service and database service is stored in the consul
        # key/value store as node_id:db_id
        self._consul_kv_store.put(node_params.id, node_params.db_id)

        # global nodes do not contain primary data(pathologies..)
        if db_params.pathologies:
            # The mapping between a database service and the primary data(pathologies)
            # it contains are dstored in the cosnul key/value store
            self._consul_kv_store.put(db_params.id, db_params.pathologies.json())

    def _check_params(self, node_params: "NodeParams", db_params: "DBParams"):
        if node_params.db_id != db_params.id:
            raise DBParamsIdNotMatchingNodeDBId(node_params.db_id, db_params.id)

        if node_params.role == NodeRole.GLOBALNODE:
            self._check_global_db_params(db_params)

        if node_params.role == NodeRole.LOCALNODE:
            self._check_local_db_params(db_params)

    def _check_global_db_params(self, db_params: "DBParams"):
        if db_params.pathologies != None:
            raise GlobalNodeCannotContainPrimaryData(db_params)

    def _check_local_db_params(self, db_params: "DBParams"):
        if db_params.pathologies == None:
            raise LocalNodeMustContainPrimaryData(db_params)

    def deregister_node(self, node_id: str):
        _, data = self._consul_kv_store.get(node_id)

        if not data:
            raise NodeIDUnknown(node_id)

        db_id = str(data["Value"], "UTF-8")

        # deregister db service
        self._consul_service.deregister(service_id=db_id)
        self._consul_kv_store.delete(db_id)

        # deregister node service
        self._consul_service.deregister(service_id=node_id)

        # delete node configuration from kv store
        self._consul_kv_store.delete(node_id)

    def get_all_nodes(self) -> List["NodeParams"]:
        all_services = self._consul_agent.services()
        all_nodes = []
        for (node_service_id, node_service_info) in all_services.items():
            tags = node_service_info["Tags"]
            role = ""
            if NodeRole.GLOBALNODE in tags:
                role = NodeRole.GLOBALNODE
            elif NodeRole.LOCALNODE in tags:
                role = NodeRole.LOCALNODE
            else:
                continue

            _, data = self._consul_kv_store.get(node_service_id)
            db_id = str(data["Value"], "UTF-8")

            node_params = NodeParams(
                id=node_service_id,
                ip=node_service_info["Address"],
                port=node_service_info["Port"],
                role=role,
                db_id=db_id,
            )
            all_nodes.append(node_params)

        return all_nodes

    def get_all_dbs(self) -> List["DBParams"]:
        all_services = self._consul_agent.services()
        all_dbs = []
        for (service_id, service_info) in all_services.items():
            tags = service_info["Tags"]
            if "db" in tags:
                db_id = service_id
                pathologies = self.get_pathologies_by_db_id(db_id)
                db_params = DBParams(
                    id=db_id,
                    ip=service_info["Address"],
                    port=service_info["Port"],
                    pathologies=pathologies,
                )
                all_dbs.append(db_params)
        return all_dbs

    def get_node_by_node_id(self, node_id):
        all_nodes = self.get_all_nodes()
        for node in all_nodes:
            if node.id == node_id:
                return node

        raise NodeIDUnknown(node_id)

    def get_all_global_nodes(self):
        all_nodes = self.get_all_nodes()
        return [
            node_params
            for node_params in all_nodes
            if node_params.role == NodeRole.GLOBALNODE
        ]

    def get_all_local_nodes(self) -> List["NodeParams"]:
        all_nodes = self.get_all_nodes()
        return [
            node_params
            for node_params in all_nodes
            if node_params.role == NodeRole.LOCALNODE
        ]

    def get_node_id_by_db_id(self, db_id: str) -> str:
        all_nodes = self.get_all_nodes()
        for node in all_nodes:
            if node.db_id == db_id:
                return node.id

        raise DBIdUnknown(db_id)

    def get_nodes_with_all_of_pathologies(
        self, pathologies: List[str]
    ) -> List["NodeParams"]:
        all_local_nodes = self.get_all_local_nodes()

        local_nodes_with_pathologies = []
        for node_params in all_local_nodes:
            db_id = node_params.db_id

            db_pathologies = self.get_pathologies_by_db_id(db_id)
            db_pathologies_list = [
                db_p.name for db_p in db_pathologies.pathologies_list
            ]
            if all(p in db_pathologies_list for p in pathologies):
                local_nodes_with_pathologies.append(node_params)

        return local_nodes_with_pathologies

    def get_nodes_with_any_of_datasets(self, datasets: List[str]) -> List["NodeParams"]:
        all_local_nodes = self.get_all_local_nodes()
        local_nodes_with_datasets = []
        for node_params in all_local_nodes:
            db_id = node_params.db_id
            db_datasets = self.get_datasets_by_db_id(db_id)
            if _compare_unordered(db_datasets, datasets):
                local_nodes_with_datasets.append(node_params)
        return local_nodes_with_datasets

    def get_db_by_node_id(self, node_id: str) -> "DBParams":
        _, data = self._consul_kv_store.get(node_id, index=None)
        if not data:
            raise NodeIdNotFoundInKVStore(node_id)
        db_id = str(data["Value"], "UTF-8")
        db_params = self.get_db_by_db_id(db_id)
        return db_params

    def get_db_by_db_id(self, db_id: str) -> "DBParams":
        all_dbs = self.get_all_dbs()
        for db in all_dbs:
            if db.id == db_id:
                return db
        raise DBIdUnknown(db_id)

    def get_pathologies_by_db_id(self, db_id: str) -> "Pathologies":
        _, data = self._consul_kv_store.get(db_id, index=None)
        if data:
            pathologies_json = data["Value"]
            pathologies = Pathologies.parse_raw(pathologies_json)
            return pathologies

    def get_datasets_by_db_id(self, db_id: str) -> List[str]:
        pathologies = self.get_pathologies_by_db_id(db_id)
        datasets = []
        for pathology in pathologies.pathologies_list:
            for dataset in pathology.datasets:
                datasets.append(dataset)
        return datasets

    def pathology_exists(self, pathology: str):
        if self.get_nodes_with_all_of_pathologies([pathology]):
            return True
        return False

    def dataset_exists(self, pathology: str, dataset: str):
        nodes = self.get_nodes_with_all_of_pathologies([pathology])
        for node in nodes:
            datasets = self.get_datasets_by_db_id(node.db_id)
            if dataset in datasets:
                return True
        return False


def _compare_unordered(a: List[Any], b: List[Any]):
    a_set = set(a)
    b_set = set(b)
    if len(a_set.intersection(b_set)) > 0:
        return True
    return False


class DBParamsIdNotMatchingNodeDBId(Exception):
    def __init__(self, node_params_db_id: str, db_params_db_id):
        self.message = f"NodeParams db_id:{node_params_db_id} not matching DBParams id:{db_params_db_id}"


class NodeIdNotFoundInKVStore(Exception):
    def __init__(self, node_id: str):
        self.message = f"There is no node_id:{node_id} in the key/value store"


class NodeIDUnknown(Exception):
    def __init__(self, node_id: str):
        self.message = f"There is no Node service with node_id:{node_id} registered"


class DBIdUnknown(Exception):
    def __init__(self, db_id: str):
        self.message = f"There is no registered DB service with db_id:{db_id}"


class GlobalNodeCannotContainPrimaryData(Exception):
    def __init__(self, db_params: "DBParams"):
        self.message = f"database parameters:\n {db_params} \ncannot be paired with a GLOBAL node because it contains primary data(pathologies)"


class LocalNodeMustContainPrimaryData(Exception):
    def __init__(self, db_params: "DBParams"):
        self.message = f"database parameters:\n {db_params} \ncannot be paired with a LOCAL node because it does contain primary data(pathologies)"


class Pathology(BaseModel):
    name: str
    datasets: List[str]


class Pathologies(BaseModel):
    pathologies_list: List["Pathology"]


@unique
class NodeRole(str, Enum):
    GLOBALNODE = "GLOBALNODE"
    LOCALNODE = "LOCALNODE"


class NodeParams(BaseModel):
    id: str
    ip: IPv4Address
    port: int
    role: "NodeRole"
    db_id: str


class DBParams(BaseModel):
    id: str
    ip: IPv4Address
    port: int
    pathologies: Optional["Pathologies"]
