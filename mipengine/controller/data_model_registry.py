from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel

from mipengine.node_tasks_DTOs import CommonDataElements


def _have_common_elements(a: List[Any], b: List[Any]):
    return bool(set(a) & set(b))


class DataModelCDES(BaseModel):
    """
    A dictionary representation of the cdes of each data model.
    Key values are data models.
    Values are CommonDataElements.
    """

    values: Optional[Dict[str, CommonDataElements]] = {}


class DatasetsLocations(BaseModel):
    """
    A dictionary representation of the locations of each dataset in the federation.
    Key values are data models because a dataset may be available in multiple data_models.
    Values are Dictionaries of datasets and their locations.
    """

    values: Optional[Dict[str, Dict[str, str]]] = {}


class DataModelRegistry(BaseModel):
    data_models: Optional[DataModelCDES] = DataModelCDES()
    datasets_locations: Optional[DatasetsLocations] = DatasetsLocations()

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True

    def get_cdes_specific_data_model(self, data_model) -> CommonDataElements:
        return self.data_models.values[data_model]

    def get_all_available_datasets_per_data_model(self) -> Dict[str, List[str]]:
        """
        Returns a dictionary with all the currently available data_models on the
        system as keys and lists of datasets as values. Without duplicates
        """
        return (
            {
                data_model: list(datasets_and_locations_of_specific_data_model.keys())
                for data_model, datasets_and_locations_of_specific_data_model in self.datasets_locations.values.items()
            }
            if self.datasets_locations
            else {}
        )

    def data_model_exists(self, data_model: str) -> bool:
        return data_model in self.datasets_locations.values

    def dataset_exists(self, data_model: str, dataset: str) -> bool:
        return (
            data_model in self.datasets_locations.values
            and dataset in self.datasets_locations.values[data_model]
        )

    def get_node_ids_with_any_of_datasets(
        self, data_model: str, datasets: List[str]
    ) -> List[str]:
        if not self.data_model_exists(data_model):
            return []

        local_nodes_with_datasets = [
            self.datasets_locations.values[data_model][dataset]
            for dataset in self.datasets_locations.values[data_model]
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
        some, all or none of bthe wanted_datasets that are located in the node
        """
        if not self.data_model_exists(data_model):
            raise ValueError(
                f"Data model '{data_model}' is not available in the node '{node_id}'."
            )

        datasets_in_node = [
            dataset
            for dataset in self.datasets_locations.values[data_model]
            if dataset in wanted_datasets
            and node_id == self.datasets_locations.values[data_model][dataset]
        ]
        return datasets_in_node
