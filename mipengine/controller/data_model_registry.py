from typing import Any
from typing import Dict
from typing import List

from mipengine.node_tasks_DTOs import CommonDataElements


def _have_common_elements(a: List[Any], b: List[Any]):
    return bool(set(a) & set(b))


class DataModelRegistry:
    def __init__(self):
        self.data_models: Dict[str, CommonDataElements] = {}
        self.datasets_location: Dict[str, Dict[str, str]] = {}

    def set_data_models(self, data_models: Dict[str, CommonDataElements]):
        self.data_models = data_models

    def set_datasets_location(self, datasets_location: Dict[str, Dict[str, str]]):
        self.datasets_location = datasets_location

    def get_all_available_datasets_per_data_model(self) -> Dict[str, List[str]]:
        """
        Returns a dictionary with all the currently available data_models on the
        system as keys and lists of datasets as values. Without duplicates
        """
        return {
            data_model: list(self.datasets_location[data_model].keys())
            for data_model in self.data_models
        }

    def data_model_exists(self, data_model: str) -> bool:
        return data_model in self.datasets_location

    def dataset_exists(self, data_model: str, dataset: str) -> bool:
        return (
            data_model in self.datasets_location
            and dataset in self.datasets_location[data_model]
        )

    def get_node_ids_with_any_of_datasets(
        self, data_model: str, datasets: List[str]
    ) -> List[str]:
        if not self.data_model_exists(data_model):
            return []

        local_nodes_with_datasets = [
            node
            for dataset in self.datasets_location[data_model]
            if dataset in datasets
            for node in self.datasets_location[data_model][dataset]
        ]
        return list(set(local_nodes_with_datasets))


data_model_registry = DataModelRegistry()
