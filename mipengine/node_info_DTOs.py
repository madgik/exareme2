from ipaddress import IPv4Address
from typing import Dict
from typing import Optional, List
from pydantic import BaseModel

from enum import Enum, unique


@unique
class NodeRole(str, Enum):
    GLOBALNODE = "GLOBALNODE"
    LOCALNODE = "LOCALNODE"


class NodeInfo(BaseModel):
    id: str
    role: NodeRole
    ip: IPv4Address
    port: int
    db_ip: IPv4Address
    db_port: int
    datasets_per_schema: Optional[Dict[str, List[str]]]
