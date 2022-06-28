from logging import Logger
from typing import Any
from typing import Dict
from typing import List

from mipengine.controller.federation_info_logs import log_datamodel_added
from mipengine.controller.federation_info_logs import log_datamodel_removed
from mipengine.controller.federation_info_logs import log_dataset_added
from mipengine.controller.federation_info_logs import log_dataset_removed
from mipengine.node_tasks_DTOs import CommonDataElement
from mipengine.node_tasks_DTOs import CommonDataElements


def _have_common_elements(a: List[Any], b: List[Any]):
    return bool(set(a) & set(b))


class DataModelRegistry:
    def __init__(self, logger: Logger):
        self._logger = logger
        self._data_models: Dict[str, CommonDataElements] = {}
        self._datasets_location: Dict[str, Dict[str, str]] = {}

    @property
    def data_models(self):
        return self._data_models

    @property
    def dataset_locations(self):
        return self._datasets_location

    @data_models.setter
    def data_models(self, value):
        _log_data_model_changes(self._logger, self._data_models, value)
        self._data_models = value

    @dataset_locations.setter
    def dataset_locations(self, value):
        _log_dataset_changes(self._logger, self._datasets_location, value)
        self._datasets_location = value

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


def _log_data_model_changes(logger, old_data_models, new_data_models):
    added_data_models = new_data_models.keys() - old_data_models.keys()
    for data_model in added_data_models:
        log_datamodel_added(data_model, logger)

    removed_data_models = old_data_models.keys() - new_data_models.keys()
    for data_model in removed_data_models:
        log_datamodel_removed(data_model, logger)


def _log_dataset_changes(
    logger, old_datasets_per_data_model, new_datasets_per_data_model
):
    _log_datasets_added(
        logger, old_datasets_per_data_model, new_datasets_per_data_model
    )
    _log_datasets_removed(
        logger, old_datasets_per_data_model, new_datasets_per_data_model
    )


def _log_datasets_added(
    logger, old_datasets_per_data_model, new_datasets_per_data_model
):
    for data_model in new_datasets_per_data_model:
        added_datasets = new_datasets_per_data_model[data_model].keys()
        if data_model in old_datasets_per_data_model:
            added_datasets -= old_datasets_per_data_model[data_model].keys()
        for dataset in added_datasets:
            log_dataset_added(data_model, dataset, logger, new_datasets_per_data_model)


def _log_datasets_removed(
    logger, old_datasets_per_data_model, new_datasets_per_data_model
):
    for data_model in old_datasets_per_data_model:
        removed_datasets = old_datasets_per_data_model[data_model].keys()
        if data_model in new_datasets_per_data_model:
            removed_datasets -= new_datasets_per_data_model[data_model].keys()
        for dataset in removed_datasets:
            log_dataset_removed(
                data_model, dataset, logger, old_datasets_per_data_model
            )
