import threading
import time
import traceback
from abc import ABC
from logging import Logger
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from pydantic import BaseModel

from exareme2.controller import DeploymentType
from exareme2.controller.celery.app import CeleryConnectionError
from exareme2.controller.celery.app import CeleryTaskTimeoutException
from exareme2.controller.federation_info_logs import log_datamodel_added
from exareme2.controller.federation_info_logs import log_datamodel_removed
from exareme2.controller.federation_info_logs import log_dataset_added
from exareme2.controller.federation_info_logs import log_dataset_removed
from exareme2.controller.federation_info_logs import log_worker_joined_federation
from exareme2.controller.federation_info_logs import log_worker_left_federation
from exareme2.controller.services.worker_landscape_aggregator.worker_info_tasks_handler import (
    WorkerInfoTasksHandler,
)
from exareme2.controller.workers_addresses import WorkersAddressesFactory
from exareme2.utils import AttrDict
from exareme2.worker_communication import CommonDataElement
from exareme2.worker_communication import CommonDataElements
from exareme2.worker_communication import DataModelAttributes
from exareme2.worker_communication import WorkerInfo
from exareme2.worker_communication import WorkerRole

WORKER_LANDSCAPE_AGGREGATOR_REQUEST_ID = "WORKER_LANDSCAPE_AGGREGATOR"
LONGITUDINAL = "longitudinal"


class ImmutableBaseModel(BaseModel, ABC):
    class Config:
        allow_mutation = False


def _get_worker_socket_addr(worker_info: WorkerInfo):
    return f"{worker_info.ip}:{worker_info.port}"


def _have_common_elements(a: List[Any], b: List[Any]):
    return bool(set(a) & set(b))


class DataModelsCDES(ImmutableBaseModel):
    """
    A dictionary representation of the cdes of each data model.
    Key values are data models.
    Values are CommonDataElements.
    """

    data_models_cdes: Optional[Dict[str, CommonDataElements]] = {}


class DataModelsAttributes(ImmutableBaseModel):
    """
    A dictionary representation of the attributes of each data model.
    Key values are data models.
    Values are DataModelAttributes.
    """

    data_models_attributes: Optional[Dict[str, DataModelAttributes]] = {}


class DatasetsLocations(ImmutableBaseModel):
    """
    A dictionary representation of the locations of each dataset in the federation.
    Key values are data models because a dataset may be available in multiple data_models.
    Values are Dictionaries of datasets and their locations.
    """

    datasets_locations: Optional[Dict[str, Dict[str, str]]] = {}


class DataModelRegistry(ImmutableBaseModel):
    data_models_attributes: Optional[DataModelsAttributes] = DataModelsAttributes()
    data_models_cdes: Optional[DataModelsCDES] = DataModelsCDES()
    datasets_locations: Optional[DatasetsLocations] = DatasetsLocations()

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True

    def is_longitudinal(self, data_model: str) -> bool:
        return (
            LONGITUDINAL
            in self.data_models_attributes.data_models_attributes[data_model].tags
        )

    def get_cdes_specific_data_model(self, data_model) -> CommonDataElements:
        return self.data_models_cdes.data_models_cdes[data_model]

    def get_data_models_attributes(self) -> Dict[str, DataModelAttributes]:
        return self.data_models_attributes.data_models_attributes

    def get_all_available_datasets_per_data_model(self) -> Dict[str, List[str]]:
        """
        Returns a dictionary with all the currently available data_models on the
        system as keys and lists of datasets as values. Without duplicates
        """
        return (
            {
                data_model: list(datasets_and_locations_of_specific_data_model.keys())
                for data_model, datasets_and_locations_of_specific_data_model in self.datasets_locations.datasets_locations.items()
            }
            if self.datasets_locations
            else {}
        )

    def data_model_exists(self, data_model: str) -> bool:
        return data_model in self.datasets_locations.datasets_locations

    def dataset_exists(self, data_model: str, dataset: str) -> bool:
        return (
            data_model in self.datasets_locations.datasets_locations
            and dataset in self.datasets_locations.datasets_locations[data_model]
        )

    def get_worker_ids_with_any_of_datasets(
        self, data_model: str, datasets: List[str]
    ) -> List[str]:
        if not self.data_model_exists(data_model):
            return []

        local_workers_with_datasets = [
            self.datasets_locations.datasets_locations[data_model][dataset]
            for dataset in self.datasets_locations.datasets_locations[data_model]
            if dataset in datasets
        ]
        return list(set(local_workers_with_datasets))

    def get_worker_specific_datasets(
        self, worker_id: str, data_model: str, wanted_datasets: List[str]
    ) -> List[str]:
        """
        From the datasets provided, returns only the ones located in the worker.

        Parameters
        ----------
        worker_id: the id of the worker
        data_model: the data model of the datasets
        wanted_datasets: the datasets to look for

        Returns
        -------
        some, all or none of the wanted_datasets that are located in the worker
        """
        if not self.data_model_exists(data_model):
            raise ValueError(
                f"Data model '{data_model}' is not available in the worker '{worker_id}'."
            )

        datasets_in_worker = [
            dataset
            for dataset in self.datasets_locations.datasets_locations[data_model]
            if dataset in wanted_datasets
            and worker_id
            == self.datasets_locations.datasets_locations[data_model][dataset]
        ]
        return datasets_in_worker


class WorkerRegistry(ImmutableBaseModel):
    def __init__(self, workers_info=None):
        if workers_info:
            global_workers = [
                worker_info
                for worker_info in workers_info
                if worker_info.role == WorkerRole.GLOBALWORKER
            ]
            local_workers = [
                worker_info
                for worker_info in workers_info
                if worker_info.role == WorkerRole.LOCALWORKER
            ]
            _workers_per_id = {
                worker_info.id: worker_info
                for worker_info in global_workers + local_workers
            }
            super().__init__(
                global_workers=global_workers,
                local_workers=local_workers,
                workers_per_id=_workers_per_id,
            )
        else:
            super().__init__()

    global_workers: List[WorkerInfo] = []
    local_workers: List[WorkerInfo] = []
    workers_per_id: Dict[str, WorkerInfo] = {}

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True


class _wlaRegistries(ImmutableBaseModel):
    worker_registry: Optional[WorkerRegistry] = WorkerRegistry()
    data_model_registry: Optional[DataModelRegistry] = DataModelRegistry()

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True


class DatasetsLabels(ImmutableBaseModel):
    """
    A dictionary representation of a dataset's information.
    Key values are the names of the datasets.
    Values are the labels of the datasets.
    """

    datasets_labels: Dict[str, str]


class DatasetsLabelsPerDataModel(ImmutableBaseModel):
    """
    Key values are the names of the data_models.
    Values are DatasetsLabels.
    """

    datasets_labels_per_data_model: Dict[str, DatasetsLabels]


class DataModelMetadata(ImmutableBaseModel):
    """
    A representation of a data model's Metadata datasets info, cdes and attributes for a specific data model
    """

    datasets_labels: DatasetsLabels
    cdes: Optional[CommonDataElements]
    attributes: Optional[DataModelAttributes]


class DataModelsMetadata(ImmutableBaseModel):
    """
    A dictionary representation of a data model's Metadata.
    Key values are data models.
    Values are DataModelMetadata.
    """

    data_models_metadata: Dict[str, DataModelMetadata]


class DataModelsMetadataPerWorker(ImmutableBaseModel):
    """
    A dictionary representation of all information for the data model's Metadata per worker.
    Key values are workers.
    Values are data model's Metadata.
    """

    data_models_metadata_per_worker: Dict[str, DataModelsMetadata]


class WorkerLandscapeAggregator:
    def __init__(
        self,
        logger: Logger,
        update_interval: int,
        tasks_timeout: int,
        run_udf_task_timeout: int,
        deployment_type: DeploymentType,
        localworkers: AttrDict,
    ):
        self._logger = logger
        self._update_interval = update_interval
        self._worker_info_tasks_timeout = run_udf_task_timeout + tasks_timeout
        self._deployment_type = deployment_type
        self._localworkers = localworkers
        self._registries = _wlaRegistries()
        self._keep_updating = True
        self._update_loop_thread = None

    def update(self):
        """
        Worker Landscape Aggregator(wla) is a module that handles the aggregation of necessary information,
        to keep up-to-date and in sync the Worker Registry and the Data Model Registry.
        The Worker Registry contains information about the worker such as id, ip, port etc.
        The Data Model Registry contains two types of information, data_models and datasets_locations.
        data_models contains information about the data models and their corresponding cdes.
        datasets_locations contains information about datasets and their locations(workers).
        wla periodically will send requests (get_worker_info, get_worker_datasets_per_data_model, get_data_model_cdes),
        to the workers to retrieve the current information that they contain.
        Once all information about data models and cdes is aggregated,
        any data model that is incompatible across workers will be removed.
        A data model is incompatible when the cdes across workers are not identical, except one edge case.
        The edge case is that the cdes can only contain a difference in the field of 'enumerations' in
        the cde with code 'dataset' and still be considered compatible.
        For each data model the 'enumerations' field in the cde with code 'dataset' is updated with all datasets across workers.
        Once all the information is aggregated and validated the wla will provide the information to the Worker Registry and to the Data Model Registry.
        """
        (
            workers_info,
            data_models_metadata_per_worker,
        ) = self._fetch_workers_metadata()
        worker_registry = WorkerRegistry(workers_info=workers_info)
        dmr = _crunch_data_model_registry_data(
            data_models_metadata_per_worker, self._logger
        )

        self._set_new_registries(worker_registry, dmr)

        self._logger.debug(
            f"Workers:{[worker_info.id for worker_info in self.get_workers()]}"
        )

    def _update_loop(self):
        while self._keep_updating:
            try:
                self.update()
            except Exception as exc:
                self._logger.warning(
                    f"WorkerLandscapeAggregator caught an exception but will continue to "
                    f"update {exc=}"
                )
                tr = traceback.format_exc()
                self._logger.error(tr)
            finally:
                time.sleep(self._update_interval)

    def start(self):
        self._logger.info("WorkerLandscapeAggregator starting ...")

        self.stop()
        self._keep_updating = True
        self._update_loop_thread = threading.Thread(
            target=self._update_loop, daemon=True
        )
        self._update_loop_thread.start()

        self._logger.info("WorkerLandscapeAggregator started.")

    def stop(self):
        if self._update_loop_thread and self._update_loop_thread.is_alive():
            self._keep_updating = False
            self._update_loop_thread.join()

    def healthcheck(self):
        worker_info_tasks_handlers = [
            WorkerInfoTasksHandler(
                worker_queue_addr=worker_queue_addr,
                tasks_timeout=self._worker_info_tasks_timeout,
                request_id=WORKER_LANDSCAPE_AGGREGATOR_REQUEST_ID,
            )
            for worker_queue_addr in WorkersAddressesFactory(
                self._deployment_type, self._localworkers
            )
            .get_workers_addresses()
            .socket_addresses
        ]
        for task_handler in worker_info_tasks_handlers:
            task_handler.get_healthcheck_task(False)

    def _get_workers_info(self, workers_socket_addr: List[str]) -> List[WorkerInfo]:
        worker_info_tasks_handlers = [
            WorkerInfoTasksHandler(
                worker_queue_addr=worker_queue_addr,
                tasks_timeout=self._worker_info_tasks_timeout,
                request_id=WORKER_LANDSCAPE_AGGREGATOR_REQUEST_ID,
            )
            for worker_queue_addr in workers_socket_addr
        ]
        workers_info = []
        for tasks_handler in worker_info_tasks_handlers:
            try:
                result = tasks_handler.get_worker_info_task()
                workers_info.append(result)
            except (CeleryConnectionError, CeleryTaskTimeoutException) as exc:
                # just log the exception do not reraise it
                self._logger.warning(exc)
            except Exception:
                # just log full traceback exception as error and do not reraise it
                self._logger.error(traceback.format_exc())
        return workers_info

    def _get_worker_datasets_per_data_model(
        self,
        worker_queue_addr: str,
    ) -> Dict[str, Dict[str, str]]:
        tasks_handler = WorkerInfoTasksHandler(
            worker_queue_addr=worker_queue_addr,
            tasks_timeout=self._worker_info_tasks_timeout,
            request_id=WORKER_LANDSCAPE_AGGREGATOR_REQUEST_ID,
        )
        try:
            datasets_per_data_model = (
                tasks_handler.get_worker_datasets_per_data_model_task()
            )
        except (CeleryConnectionError, CeleryTaskTimeoutException) as exc:
            # just log the exception do not reraise it
            self._logger.warning(exc)
            return {}
        except Exception:
            # just log full traceback exception as error and do not reraise it
            self._logger.error(traceback.format_exc())
            return {}
        return datasets_per_data_model

    def _get_worker_cdes(
        self, worker_queue_addr: str, data_model: str
    ) -> CommonDataElements:
        tasks_handler = WorkerInfoTasksHandler(
            worker_queue_addr=worker_queue_addr,
            tasks_timeout=self._worker_info_tasks_timeout,
            request_id=WORKER_LANDSCAPE_AGGREGATOR_REQUEST_ID,
        )
        try:
            worker_cdes = tasks_handler.get_data_model_cdes_task(
                data_model=data_model,
            )
            return worker_cdes
        except (CeleryConnectionError, CeleryTaskTimeoutException) as exc:
            # just log the exception do not reraise it
            self._logger.warning(exc)
        except Exception:
            # just log full traceback exception as error and do not reraise it
            self._logger.error(traceback.format_exc())

    def _get_data_model_attributes(
        self, worker_queue_addr: str, data_model: str
    ) -> DataModelAttributes:
        tasks_handler = WorkerInfoTasksHandler(
            worker_queue_addr=worker_queue_addr,
            tasks_timeout=self._worker_info_tasks_timeout,
            request_id=WORKER_LANDSCAPE_AGGREGATOR_REQUEST_ID,
        )
        try:
            attributes = tasks_handler.get_data_model_attributes_task(
                data_model=data_model
            )
            return attributes
        except (CeleryConnectionError, CeleryTaskTimeoutException) as exc:
            # just log the exception do not reraise it
            self._logger.warning(exc)
        except Exception:
            # just log full traceback exception as error and do not reraise it
            self._logger.error(traceback.format_exc())

    def _set_new_registries(self, worker_registry, data_model_registry):
        _log_worker_changes(
            self.get_workers(),
            list(worker_registry.workers_per_id.values()),
            self._logger,
        )
        _log_data_model_changes(
            self._registries.data_model_registry.data_models_cdes.data_models_cdes,
            data_model_registry.data_models_cdes.data_models_cdes,
            self._logger,
        )
        _log_dataset_changes(
            self._registries.data_model_registry.datasets_locations.datasets_locations,
            data_model_registry.datasets_locations.datasets_locations,
            self._logger,
        )
        self._registries = _wlaRegistries(
            worker_registry=worker_registry, data_model_registry=data_model_registry
        )

    def get_workers(self) -> List[WorkerInfo]:
        return list(self._registries.worker_registry.workers_per_id.values())

    def get_global_worker(self) -> WorkerInfo:
        if not self._registries.worker_registry.global_workers:
            raise Exception("Global Worker is unavailable")
        return self._registries.worker_registry.global_workers[0]

    def get_all_local_workers(self) -> List[WorkerInfo]:
        return self._registries.worker_registry.local_workers

    def get_worker_info(self, worker_id: str) -> WorkerInfo:
        return self._registries.worker_registry.workers_per_id[worker_id]

    def get_cdes(self, data_model: str) -> Dict[str, CommonDataElement]:
        return self._registries.data_model_registry.get_cdes_specific_data_model(
            data_model
        ).values

    def get_metadata(self, data_model: str, variable_names: List[str]):
        common_data_elements = self.get_cdes(data_model)
        metadata = {
            variable_name: cde.dict()
            for variable_name, cde in common_data_elements.items()
            if variable_name in variable_names
        }
        return metadata

    def get_cdes_per_data_model(self) -> DataModelsCDES:
        return self._registries.data_model_registry.data_models_cdes

    def get_datasets_locations(self) -> DatasetsLocations:
        return self._registries.data_model_registry.datasets_locations

    def get_all_available_datasets_per_data_model(self) -> Dict[str, List[str]]:
        return (
            self._registries.data_model_registry.get_all_available_datasets_per_data_model()
        )

    def data_model_exists(self, data_model: str) -> bool:
        return self._registries.data_model_registry.data_model_exists(data_model)

    def dataset_exists(self, data_model: str, dataset: str) -> bool:
        return self._registries.data_model_registry.dataset_exists(data_model, dataset)

    def get_worker_ids_with_any_of_datasets(
        self, data_model: str, datasets: List[str]
    ) -> List[str]:
        return self._registries.data_model_registry.get_worker_ids_with_any_of_datasets(
            data_model, datasets
        )

    def get_worker_specific_datasets(
        self, worker_id: str, data_model: str, wanted_datasets: List[str]
    ) -> List[str]:
        return self._registries.data_model_registry.get_worker_specific_datasets(
            worker_id, data_model, wanted_datasets
        )

    def get_data_models_attributes(self) -> Dict[str, DataModelAttributes]:
        return self._registries.data_model_registry.get_data_models_attributes()

    def _fetch_workers_metadata(
        self,
    ) -> Tuple[List[WorkerInfo], DataModelsMetadataPerWorker,]:
        """
        Returns a list of all the workers in the federation and their metadata (data_models, datasets, cdes).
        """
        workers_addresses = (
            WorkersAddressesFactory(self._deployment_type, self._localworkers)
            .get_workers_addresses()
            .socket_addresses
        )
        workers_info = self._get_workers_info(workers_addresses)
        local_workers = [
            worker_info
            for worker_info in workers_info
            if worker_info.role == WorkerRole.LOCALWORKER
        ]
        data_models_metadata_per_worker = self._get_data_models_metadata_per_worker(
            local_workers
        )
        return workers_info, data_models_metadata_per_worker

    def _get_data_models_metadata_per_worker(
        self,
        workers: List[WorkerInfo],
    ) -> DataModelsMetadataPerWorker:
        data_models_metadata_per_worker = {}

        for worker_info in workers:
            data_models_metadata = {}

            worker_socket_addr = _get_worker_socket_addr(worker_info)
            datasets_per_data_model = self._get_worker_datasets_per_data_model(
                worker_socket_addr
            )
            if datasets_per_data_model:
                worker_socket_addr = _get_worker_socket_addr(worker_info)
                for data_model, datasets in datasets_per_data_model.items():
                    cdes = self._get_worker_cdes(worker_socket_addr, data_model)
                    attributes = self._get_data_model_attributes(
                        worker_socket_addr, data_model
                    )
                    cdes = cdes if cdes else None
                    attributes = attributes if attributes else None
                    data_models_metadata[data_model] = DataModelMetadata(
                        datasets_labels=DatasetsLabels(datasets_labels=datasets),
                        cdes=cdes,
                        attributes=attributes,
                    )
                    data_models_metadata_per_worker[
                        worker_info.id
                    ] = DataModelsMetadata(data_models_metadata=data_models_metadata)

        return DataModelsMetadataPerWorker(
            data_models_metadata_per_worker=data_models_metadata_per_worker
        )


def _crunch_data_model_registry_data(
    data_models_metadata_per_worker: DataModelsMetadataPerWorker, logger
) -> DataModelRegistry:
    incompatible_data_models = _get_incompatible_data_models(
        data_models_metadata_per_worker, logger
    )
    data_models_metadata_per_worker_with_compatible_data_models = (
        _remove_incompatible_data_models_from_data_models_metadata_per_worker(
            data_models_metadata_per_worker, incompatible_data_models
        )
    )
    (
        datasets_locations,
        datasets_labels_per_data_model,
    ) = _aggregate_datasets_locations_and_labels(
        data_models_metadata_per_worker_with_compatible_data_models, logger
    )
    data_models_cdes = _aggregate_data_models_cdes(
        data_models_metadata_per_worker_with_compatible_data_models,
        datasets_labels_per_data_model,
    )
    data_models_attributes = _aggregate_data_models_attributes(
        data_models_metadata_per_worker_with_compatible_data_models,
    )

    return DataModelRegistry(
        data_models_cdes=data_models_cdes,
        datasets_locations=datasets_locations,
        data_models_attributes=data_models_attributes,
    )


def _aggregate_datasets_locations_and_labels(
    data_models_metadata_per_worker, logger
) -> Tuple[DatasetsLocations, DatasetsLabelsPerDataModel]:
    """
    Args:
        data_models_metadata_per_worker
    Returns:
        A tuple with:
         1. DatasetsLocations
         2. DatasetsLabelsPerDataModel
    """
    datasets_locations = {}
    datasets_labels = {}
    for (
        worker_id,
        data_models_metadata,
    ) in data_models_metadata_per_worker.data_models_metadata_per_worker.items():
        for (
            data_model,
            data_model_metadata,
        ) in data_models_metadata.data_models_metadata.items():
            current_labels = (
                datasets_labels[data_model].datasets_labels
                if data_model in datasets_labels
                else {}
            )
            current_datasets = (
                datasets_locations[data_model]
                if data_model in datasets_locations
                else {}
            )

            for (
                dataset_name,
                dataset_label,
            ) in data_model_metadata.datasets_labels.datasets_labels.items():
                current_labels[dataset_name] = dataset_label

                if dataset_name in current_datasets:
                    current_datasets[dataset_name].append(worker_id)
                else:
                    current_datasets[dataset_name] = [worker_id]

            datasets_labels[data_model] = DatasetsLabels(datasets_labels=current_labels)
            datasets_locations[data_model] = current_datasets

    datasets_locations_without_duplicates = {}
    for data_model, dataset_locations in datasets_locations.items():
        datasets_locations_without_duplicates[data_model] = {}

        for dataset, worker_ids in dataset_locations.items():
            if len(worker_ids) == 1:
                datasets_locations_without_duplicates[data_model][dataset] = worker_ids[
                    0
                ]
            else:
                del datasets_labels[data_model].datasets_labels[dataset]
                _log_duplicated_dataset(worker_ids, data_model, dataset, logger)

    return DatasetsLocations(
        datasets_locations=datasets_locations_without_duplicates
    ), DatasetsLabelsPerDataModel(datasets_labels_per_data_model=datasets_labels)


def _aggregate_data_models_cdes(
    data_models_metadata_per_worker: DataModelsMetadataPerWorker,
    datasets_labels_per_data_model: DatasetsLabelsPerDataModel,
) -> DataModelsCDES:
    data_models = {}
    for (
        worker_id,
        data_models_metadata,
    ) in data_models_metadata_per_worker.data_models_metadata_per_worker.items():
        for (
            data_model,
            data_model_metadata,
        ) in data_models_metadata.data_models_metadata.items():
            data_models[data_model] = data_model_metadata.cdes
            dataset_cde = data_model_metadata.cdes.values["dataset"]
            new_dataset_cde = CommonDataElement(
                code=dataset_cde.code,
                label=dataset_cde.label,
                sql_type=dataset_cde.sql_type,
                is_categorical=dataset_cde.is_categorical,
                enumerations=datasets_labels_per_data_model.datasets_labels_per_data_model[
                    data_model
                ].datasets_labels,
                min=dataset_cde.min,
                max=dataset_cde.max,
            )
            data_models[data_model].values["dataset"] = new_dataset_cde
    return DataModelsCDES(data_models_cdes=data_models)


def _aggregate_data_models_attributes(
    data_models_metadata_per_worker: DataModelsMetadataPerWorker,
) -> DataModelsAttributes:
    data_models_attributes = {}
    for (
        worker_id,
        data_models_metadata,
    ) in data_models_metadata_per_worker.data_models_metadata_per_worker.items():
        for (
            data_model,
            data_model_metadata,
        ) in data_models_metadata.data_models_metadata.items():
            current_tags = _get_updated_tags(
                data_model=data_model,
                data_models_attributes=data_models_attributes,
                tags=data_model_metadata.attributes.tags,
            )
            current_properties = _get_updated_properties(
                data_model=data_model,
                data_models_attributes=data_models_attributes,
                properties_to_be_added=data_model_metadata.attributes.properties,
            )
            data_models_attributes[data_model] = DataModelAttributes(
                tags=current_tags, properties=current_properties
            )
    return DataModelsAttributes(data_models_attributes=data_models_attributes)


def _get_updated_properties(data_model, data_models_attributes, properties_to_be_added):
    if data_model not in data_models_attributes:
        return {key: [value] for key, value in properties_to_be_added.items()}

    properties = data_models_attributes[data_model].properties
    for key, value in properties_to_be_added.items():
        properties[key] = (
            (properties[key] if value in properties[key] else properties[key] + [value])
            if key in properties
            else [value]
        )
    return properties


def _get_updated_tags(data_model, data_models_attributes, tags):
    return (
        data_models_attributes[data_model].tags
        + list(set(tags) - set(data_models_attributes[data_model].tags))
        if data_model in data_models_attributes
        else tags
    )


def _get_incompatible_data_models(
    data_models_metadata_per_worker: DataModelsMetadataPerWorker, logger
) -> List[str]:
    """
    Each worker has its own data models definition.
    We need to check for each data model if the definitions across all workers is the same.
    If the data model is not the same across all workers containing it, we log the incompatibility.
    The data models with similar definitions across all workers are returned.
    Parameters
    ----------
        data_models_metadata_per_worker: DataModelsMetadataPerWorker
    Returns
    ----------
        List[str]
            The incompatible data models
    """
    validation_dictionary = {}

    incompatible_data_models = []
    for (
        worker_id,
        data_models_metadata,
    ) in data_models_metadata_per_worker.data_models_metadata_per_worker.items():
        for (
            data_model,
            data_model_metadata,
        ) in data_models_metadata.data_models_metadata.items():
            if (
                data_model in incompatible_data_models
                or not data_model_metadata.cdes
                or not data_model_metadata.attributes
            ):
                continue

            if data_model in validation_dictionary:
                valid_worker_id, valid_cdes = validation_dictionary[data_model]
                if not valid_cdes == data_model_metadata.cdes:
                    workers = [worker_id, valid_worker_id]
                    incompatible_data_models.append(data_model)
                    _log_incompatible_data_models(
                        workers,
                        data_model,
                        [data_model_metadata.cdes, valid_cdes],
                        logger,
                    )
                    break
            else:
                validation_dictionary[data_model] = (
                    worker_id,
                    data_model_metadata.cdes,
                )

    return incompatible_data_models


def _remove_incompatible_data_models_from_data_models_metadata_per_worker(
    data_models_metadata_per_worker: DataModelsMetadataPerWorker,
    incompatible_data_models: List[str],
) -> DataModelsMetadataPerWorker:
    return DataModelsMetadataPerWorker(
        data_models_metadata_per_worker={
            worker_id: DataModelsMetadata(
                data_models_metadata={
                    data_model: data_model_metadata
                    for data_model, data_model_metadata in data_models_metadata.data_models_metadata.items()
                    if data_model not in incompatible_data_models
                    and data_model_metadata.cdes
                    and data_model_metadata.attributes
                }
            )
            for worker_id, data_models_metadata in data_models_metadata_per_worker.data_models_metadata_per_worker.items()
        }
    )


def _log_worker_changes(old_workers, new_workers, logger):
    old_workers_per_worker_id = {worker.id: worker for worker in old_workers}
    new_workers_per_worker_id = {worker.id: worker for worker in new_workers}
    added_workers = set(new_workers_per_worker_id.keys()) - set(
        old_workers_per_worker_id.keys()
    )
    for worker in added_workers:
        log_worker_joined_federation(logger, worker)
    removed_workers = set(old_workers_per_worker_id.keys()) - set(
        new_workers_per_worker_id.keys()
    )
    for worker in removed_workers:
        log_worker_left_federation(logger, worker)


def _log_data_model_changes(old_data_models, new_data_models, logger):
    added_data_models = new_data_models.keys() - old_data_models.keys()
    for data_model in added_data_models:
        log_datamodel_added(data_model, logger)
    removed_data_models = old_data_models.keys() - new_data_models.keys()
    for data_model in removed_data_models:
        log_datamodel_removed(data_model, logger)


def _log_dataset_changes(old_datasets_locations, new_datasets_locations, logger):
    _log_datasets_added(old_datasets_locations, new_datasets_locations, logger)
    _log_datasets_removed(old_datasets_locations, new_datasets_locations, logger)


def _log_datasets_added(old_datasets_locations, new_datasets_locations, logger):
    for data_model in new_datasets_locations:
        added_datasets = new_datasets_locations[data_model].keys()
        if data_model in old_datasets_locations:
            added_datasets -= old_datasets_locations[data_model].keys()
        for dataset in added_datasets:
            log_dataset_added(
                data_model,
                dataset,
                logger,
                new_datasets_locations[data_model][dataset],
            )


def _log_datasets_removed(old_datasets_locations, new_datasets_locations, logger):
    for data_model in old_datasets_locations:
        removed_datasets = old_datasets_locations[data_model].keys()
        if data_model in new_datasets_locations:
            removed_datasets -= new_datasets_locations[data_model].keys()
        for dataset in removed_datasets:
            log_dataset_removed(
                data_model,
                dataset,
                logger,
                old_datasets_locations[data_model][dataset],
            )


def _log_incompatible_data_models(workers, data_model, conflicting_cdes, logger):
    logger.info(
        f"""Workers: '[{", ".join(workers)}]' on data model '{data_model}' have incompatibility on the following cdes: '[{", ".join([cdes.__str__() for cdes in conflicting_cdes])}]' """
    )


def _log_duplicated_dataset(workers, data_model, dataset, logger):
    logger.info(
        f"""Dataset '{dataset}' of the data_model: '{data_model}' is not unique in the federation. Workers that contain the dataset: '[{", ".join(workers)}]'"""
    )
