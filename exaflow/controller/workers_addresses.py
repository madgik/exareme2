import json
from abc import ABC
from abc import abstractmethod
from itertools import chain
from typing import List

import dns.resolver

from exaflow.controller import DeploymentType


class WorkersAddresses(ABC):
    @abstractmethod
    def __init__(self):
        self._socket_addresses = None

    @property
    def socket_addresses(self) -> List[str]:
        return self._socket_addresses


class LocalWorkersAddresses(WorkersAddresses):
    def __init__(self, localworkers_configs):
        with open(localworkers_configs.config_file) as fp:
            self._socket_addresses = json.load(fp)


class DNSWorkersAddresses(WorkersAddresses):
    def __init__(self, localworkers_configs):
        dns_names = [
            n.strip() for n in localworkers_configs.dns.split(",") if n.strip()
        ]
        ip_lists = (dns.resolver.resolve(name, "A", search=True) for name in dns_names)

        # flatten the iterator‑of‑iterators and de‑duplicate
        uniq_ips = {rr.address for rr in chain.from_iterable(ip_lists)}
        self._socket_addresses = [
            f"{ip}:{localworkers_configs.port}" for ip in uniq_ips
        ]


class WorkersAddressesFactory:
    def __init__(self, depl_type: DeploymentType, localworkers_configs):
        self.depl_type = depl_type
        self.localworkers_configs = localworkers_configs

    def get_workers_addresses(self) -> WorkersAddresses:
        if self.depl_type == DeploymentType.LOCAL:
            return LocalWorkersAddresses(self.localworkers_configs)

        if self.depl_type == DeploymentType.KUBERNETES:
            return DNSWorkersAddresses(self.localworkers_configs)

        raise ValueError(
            f"DeploymentType can be one of the following: {[t.value for t in DeploymentType]}, "
            f"value provided: '{self.depl_type}'"
        )
