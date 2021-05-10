from pydantic import BaseModel
from ipaddress import IPv4Address
from typing import Optional, List


class Pathology(BaseModel):
    name: str
    datasets: List[str]


class NodeRecord(BaseModel):
    node_id: str
    task_queue_ip: IPv4Address
    task_queue_port: int

    db_ip: IPv4Address
    db_queue_port: int

    pathologies: Optional[List[Pathology]]
