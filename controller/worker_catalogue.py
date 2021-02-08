import importlib.resources as pkg_resources
import json
from dataclasses import dataclass
from typing import Dict, List

from dataclasses_json import dataclass_json

from controller import config
from controller.utils import Singleton


@dataclass_json
@dataclass
class Worker:
    ip: str
    port: int
    data: Dict[str, List[str]]


@dataclass_json
@dataclass
class WorkerCatalogue(metaclass=Singleton):
    workers: Dict[str, Worker]
    pathologies: Dict[str, List[str]]

    def __init__(self):
        worker_catalogue_str = pkg_resources.read_text(config, 'worker_catalogue.json')
        workers_dict: Dict[str, Worker] = json.loads(worker_catalogue_str)
        self.workers = {worker_name: Worker.from_dict(worker_values)
                        for worker_name, worker_values in workers_dict.items()}
        self.pathologies = {}
        for worker in self.workers.values():
            for pathology_name, datasets in worker.data.items():
                if pathology_name not in self.pathologies.keys():
                    self.pathologies[pathology_name] = datasets
                else:
                    self.pathologies[pathology_name].extend(datasets)


WorkerCatalogue()
