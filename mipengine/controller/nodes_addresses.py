import json
from abc import ABC
from abc import abstractmethod
from typing import List

import dns.resolver

from mipengine.controller import DeploymentType


class NodesAddresses(ABC):
    @abstractmethod
    def __init__(self):
        self._socket_addresses = None

    @property
    def socket_addresses(self) -> List[str]:
        return self._socket_addresses


class LocalNodesAddresses(NodesAddresses):
    def __init__(self, localnodes):
        with open(localnodes.config_file) as fp:
            self._socket_addresses = json.load(fp)


class DNSNodesAddresses(NodesAddresses):
    def __init__(self, localnodes):
        localnode_ips = dns.resolver.resolve(localnodes.dns, "A", search=True)
        self._socket_addresses = [f"{ip}:{localnodes.port}" for ip in localnode_ips]


class NodesAddressesFactory:
    def __init__(self, depl_type: DeploymentType, localnodes):
        self.depl_type = depl_type
        self.localnodes = localnodes

    def get_nodes_addresses(self) -> NodesAddresses:
        if self.depl_type == DeploymentType.LOCAL:
            return LocalNodesAddresses(self.localnodes)

        if self.depl_type == DeploymentType.KUBERNETES:
            return DNSNodesAddresses(self.localnodes)

        raise ValueError(
            f"DeploymentType can be one of the following: {[t.value for t in DeploymentType]}, "
            f"value provided: '{self.depl_type}'"
        )
