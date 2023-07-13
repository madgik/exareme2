import json
from abc import ABC
from abc import abstractmethod
from typing import List

import dns.resolver

from exareme2.controller import DeploymentType


class NodesAddresses(ABC):
    @abstractmethod
    def __init__(self):
        self._socket_addresses = None

    @property
    def socket_addresses(self) -> List[str]:
        return self._socket_addresses


class LocalNodesAddresses(NodesAddresses):
    def __init__(self, localnodes_configs):
        with open(localnodes_configs.config_file) as fp:
            self._socket_addresses = json.load(fp)


class DNSNodesAddresses(NodesAddresses):
    def __init__(self, localnodes_configs):
        localnode_ips = dns.resolver.resolve(localnodes_configs.dns, "A", search=True)
        self._socket_addresses = [
            f"{ip}:{localnodes_configs.port}" for ip in localnode_ips
        ]


class NodesAddressesFactory:
    def __init__(self, depl_type: DeploymentType, localnodes_configs):
        self.depl_type = depl_type
        self.localnodes_configs = localnodes_configs

    def get_nodes_addresses(self) -> NodesAddresses:
        if self.depl_type == DeploymentType.LOCAL:
            return LocalNodesAddresses(self.localnodes_configs)

        if self.depl_type == DeploymentType.KUBERNETES:
            return DNSNodesAddresses(self.localnodes_configs)

        raise ValueError(
            f"DeploymentType can be one of the following: {[t.value for t in DeploymentType]}, "
            f"value provided: '{self.depl_type}'"
        )
