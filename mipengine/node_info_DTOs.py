from enum import Enum
from enum import unique
from ipaddress import IPv4Address

from pydantic import BaseModel


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
