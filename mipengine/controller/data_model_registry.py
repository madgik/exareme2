from logging import Logger
from typing import Any
from typing import Dict
from typing import List

from pydantic import BaseModel

from mipengine.node_tasks_DTOs import CommonDataElement
from mipengine.node_tasks_DTOs import CommonDataElements


def _have_common_elements(a: List[Any], b: List[Any]):
    return bool(set(a) & set(b))


class DataModelRegistry(BaseModel):
    data_models: Dict[str, CommonDataElements] = {}
    dataset_locations: Dict[str, Dict[str, str]] = {}

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True

    def get_cdes(self, data_model) -> Dict[str, CommonDataElement]:
        return self.data_models[data_model].values

    def get_all_available_datasets_per_data_model(self) -> Dict[str, List[str]]:
        """
        Returns a dictionary with all the currently available data_models on the
        system as keys and lists of datasets as values. Without duplicates
        """
        return {
            data_model: list(self.dataset_locations[data_model].keys())
            for data_model in self.data_models
        }

    def data_model_exists(self, data_model: str) -> bool:
        return data_model in self.dataset_locations

    def dataset_exists(self, data_model: str, dataset: str) -> bool:
        return (
            data_model in self.dataset_locations
            and dataset in self.dataset_locations[data_model]
        )

    def get_node_ids_with_any_of_datasets(
        self, data_model: str, datasets: List[str]
    ) -> List[str]:
        if not self.data_model_exists(data_model):
            return []

        local_nodes_with_datasets = [
            self.dataset_locations[data_model][dataset]
            for dataset in self.dataset_locations[data_model]
            if dataset in datasets
        ]
        return list(set(local_nodes_with_datasets))

    def get_node_specific_datasets(
        self, node_id: str, data_model: str, wanted_datasets: List[str]
    ) -> List[str]:
        """
        From the datasets provided, returns only the ones located in the node.

        Parameters
        ----------
        node_id: the id of the node
        data_model: the data model of the datasets
        wanted_datasets: the datasets to look for

        Returns
        -------
        some, all or none of the wanted_datasets that are located in the node
        """
        if not self.data_model_exists(data_model):
            raise ValueError(
                f"Data model '{data_model}' is not available in the node '{node_id}'."
            )

        datasets_in_node = [
            dataset
            for dataset in self.dataset_locations[data_model]
            if dataset in wanted_datasets
            and node_id == self.dataset_locations[data_model][dataset]
        ]
        return datasets_in_node
