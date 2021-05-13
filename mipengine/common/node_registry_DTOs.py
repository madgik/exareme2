from pydantic import BaseModel
from ipaddress import IPv4Address
from typing import Optional, List
from enum import Enum, unique


@unique
class NodeRole(Enum):
    GLOBAL_NODE = 0
    LOCAL_NODE = 1


class Pathology(BaseModel):
    name: str
    datasets: List[str]


class NodeRecord(BaseModel):
    node_id: str
    node_role: NodeRole
    task_queue_ip: IPv4Address
    task_queue_port: int

    db_id: str
    db_ip: IPv4Address
    db_port: int

    pathologies: Optional[List[Pathology]]


# class NodeRecordsList(BaseModel):
#     node_records: List[NodeRecord]
