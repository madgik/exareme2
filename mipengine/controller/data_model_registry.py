from typing import Any
from typing import Dict
from typing import List


def _have_common_elements(a: List[Any], b: List[Any]):
    return bool(set(a) & set(b))


class DataModelRegistry:
    def __init__(self):
        self.common_data_models = {}
        self.datasets_location = {}

    def set_common_data_models(self, cdes_per_data_model):
        self.common_data_models = cdes_per_data_model

    def set_datasets_location(self, datasets_location):
        self.datasets_location = datasets_location

    # returns a list of all the currently available data_models on the system
    # without duplicates
    def get_all_available_data_models(self) -> List[str]:
        return list(self.common_data_models.keys())

    # returns a dictionary with all the currently available data_models on the
    # system as keys and lists of datasets as values. Without duplicates
    def get_all_available_datasets_per_data_model(self) -> Dict[str, List[str]]:
        return {
            data_model: list(self.datasets_location[data_model].keys())
            for data_model in self.common_data_models
        }

    def data_model_exists(self, data_model: str):
        return data_model in self.datasets_location

    def dataset_exists(self, data_model: str, dataset: str):
        return (
            data_model in self.datasets_location
            and dataset in self.datasets_location[data_model]
        )

    def get_nodes_with_any_of_datasets(
        self, data_model: str, datasets: List[str]
    ) -> List[str]:
        local_nodes_with_datasets = []
        if data_model not in self.datasets_location:
            return []

        for dataset in self.datasets_location[data_model]:
            if dataset not in datasets:
                continue
            for node in self.datasets_location[data_model][dataset]:
                if node in local_nodes_with_datasets:
                    continue
                local_nodes_with_datasets.append(node)
        return local_nodes_with_datasets


data_model_registry = DataModelRegistry()
