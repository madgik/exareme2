import json
from abc import ABC
from abc import abstractmethod
from typing import List

import dns.resolver

from mipengine.controller import DeploymentType
from mipengine.controller import config as controller_config


class NodesAddresses(ABC):
    @abstractmethod
    def __init__(self):
        self._socket_addresses = None

    @property
    def socket_addresses(self) -> List[str]:
        return self._socket_addresses


class LocalNodesAddresses(NodesAddresses):
    def __init__(self):
        with open(controller_config.localnodes.config_file) as fp:
            self._socket_addresses = json.load(fp)


class DNSNodesAddresses(NodesAddresses):
    def __init__(self):
        localnode_ips = dns.resolver.resolve(
            controller_config.localnodes.dns, "A", search=True
        )
        self._socket_addresses = [
            f"{ip}:{controller_config.localnodes.port}" for ip in localnode_ips
        ]


class NodesAddressesFactory:
    def __init__(self, depl_type: DeploymentType):
        self.depl_type = depl_type

    def get_nodes_addresses(self) -> NodesAddresses:
        if self.depl_type == DeploymentType.LOCAL:
            return LocalNodesAddresses()

        if self.depl_type == DeploymentType.KUBERNETES:
            return DNSNodesAddresses()

        raise ValueError(
            f"DeploymentType can be one of the following: {[t.value for t in DeploymentType]}, "
            f"value provided: '{self.depl_type}'"
        )
