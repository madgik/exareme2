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
from exareme2.worker_communication import DataModelMetadata
from exareme2.worker_communication import WorkerInfo
from exareme2.worker_communication import WorkerRole

WORKER_LANDSCAPE_AGGREGATOR_REQUEST_ID = "WORKER_LANDSCAPE_AGGREGATOR"


class ImmutableBaseModel(BaseModel, ABC):
    class Config:
        allow_mutation = False


class DataModelConfict(ImmutableBaseModel):
    worker_id: str
    metadata: DataModelMetadata


def _get_worker_socket_addr(worker_info: WorkerInfo) -> str:
    return f"{worker_info.ip}:{worker_info.port}"


def _have_common_elements(a: List[Any], b: List[Any]) -> bool:
    return bool(set(a) & set(b))


class DataModelRegistry(ImmutableBaseModel):
    data_models_metadata: Dict[str, DataModelMetadata] = {}
    datasets_locations: Dict[str, Dict[str, str]] = {}

    def is_longitudinal(self, data_model: str) -> bool:
        return self.data_models_metadata[data_model].longitudinal

    def get_cdes_specific_data_model(self, data_model: str):
        return self.data_models_metadata[data_model].flatten_variables()

    def get_data_models_metadata(self) -> Dict[str, DataModelMetadata]:
        return self.data_models_metadata

    def get_all_available_datasets_per_data_model(self) -> Dict[str, List[str]]:
        return (
            {
                data_model: list(datasets.keys())
                for data_model, datasets in self.datasets_locations.items()
            }
            if self.datasets_locations
            else {}
        )

    def data_model_exists(self, data_model: str) -> bool:
        return data_model in self.datasets_locations

    def dataset_exists(self, data_model: str, dataset: str) -> bool:
        return dataset in self.datasets_locations.get(data_model, {})

    def get_worker_ids_with_any_of_datasets(
        self, data_model: str, datasets: List[str]
    ) -> List[str]:
        if not self.data_model_exists(data_model):
            return []
        return list(
            set(
                self.datasets_locations[data_model][dataset]
                for dataset in self.datasets_locations[data_model]
                if dataset in datasets
            )
        )

    def get_worker_specific_datasets(
        self, worker_id: str, data_model: str, wanted_datasets: List[str]
    ) -> List[str]:
        if not self.data_model_exists(data_model):
            raise ValueError(
                f"Data model '{data_model}' is not available in the worker '{worker_id}'."
            )
        return [
            dataset
            for dataset in self.datasets_locations[data_model]
            if dataset in wanted_datasets
            and self.datasets_locations[data_model][dataset] == worker_id
        ]


class WorkerRegistry(ImmutableBaseModel):
    global_workers: List[WorkerInfo] = []
    local_workers: List[WorkerInfo] = []
    workers_per_id: Dict[str, WorkerInfo] = {}

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
            workers_per_id = {
                worker_info.id: worker_info
                for worker_info in global_workers + local_workers
            }
            super().__init__(
                global_workers=global_workers,
                local_workers=local_workers,
                workers_per_id=workers_per_id,
            )
        else:
            super().__init__()


class _wlaRegistries(ImmutableBaseModel):
    worker_registry: Optional[WorkerRegistry] = WorkerRegistry()
    data_model_registry: Optional[DataModelRegistry] = DataModelRegistry()


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
        worker_registry = self._fetch_worker_registry()
        dmr = self._fetch_dmr_registries(worker_registry.local_workers)
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
                    f"WorkerLandscapeAggregator caught an exception but will continue to update {exc=}"
                )
                self._logger.error(traceback.format_exc())
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
                worker_queue_addr=addr,
                tasks_timeout=self._worker_info_tasks_timeout,
                request_id=WORKER_LANDSCAPE_AGGREGATOR_REQUEST_ID,
            )
            for addr in WorkersAddressesFactory(
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
                worker_queue_addr=addr,
                tasks_timeout=self._worker_info_tasks_timeout,
                request_id=WORKER_LANDSCAPE_AGGREGATOR_REQUEST_ID,
            )
            for addr in workers_socket_addr
        ]
        workers_info = []
        for tasks_handler in worker_info_tasks_handlers:
            try:
                workers_info.append(tasks_handler.get_worker_info_task())
            except (CeleryConnectionError, CeleryTaskTimeoutException) as exc:
                self._logger.warning(exc)
            except Exception:
                self._logger.error(traceback.format_exc())
        return workers_info

    def _set_new_registries(self, worker_registry, data_model_registry):
        _log_worker_changes(
            self.get_workers(),
            list(worker_registry.workers_per_id.values()),
            self._logger,
        )
        _log_data_model_changes(
            self._registries.data_model_registry.datasets_locations,
            data_model_registry.datasets_locations,
            self._logger,
        )
        _log_dataset_changes(
            self._registries.data_model_registry.datasets_locations,
            data_model_registry.datasets_locations,
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

    def get_data_models(self) -> List[str]:
        self._logger.debug(self._registries.data_model_registry.datasets_locations)
        return list(self._registries.data_model_registry.datasets_locations.keys())

    def get_cdes(self, data_model: str) -> Dict[str, CommonDataElement]:
        return self._registries.data_model_registry.data_models_metadata[
            data_model
        ].flatten_variables()

    def get_metadata(
        self, data_model: str, variable_names: List[str]
    ) -> Dict[str, Dict]:
        common_data_elements = self.get_cdes(data_model)
        return {
            variable_name: cde.to_dict()
            for variable_name, cde in common_data_elements.items()
            if variable_name in variable_names
        }

    def get_cdes_per_data_model(self) -> Dict[str, Dict[str, CommonDataElement]]:
        return {
            data_model: self.get_cdes(data_model)
            for data_model in self.get_data_models()
        }

    def get_datasets_locations(self) -> Dict:
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

    def get_data_models_metadata(self) -> Dict[str, DataModelMetadata]:
        return self._registries.data_model_registry.get_data_models_metadata()

    def _fetch_worker_registry(self) -> WorkerRegistry:
        workers_addresses = (
            WorkersAddressesFactory(self._deployment_type, self._localworkers)
            .get_workers_addresses()
            .socket_addresses
        )
        return WorkerRegistry(workers_info=self._get_workers_info(workers_addresses))

    def _get_worker_data_model_metadata_and_datasets(
        self, worker_queue_addr: str
    ) -> Tuple[Dict[str, DataModelMetadata], Dict[str, List[str]]]:
        tasks_handler = WorkerInfoTasksHandler(
            worker_queue_addr=worker_queue_addr,
            tasks_timeout=self._worker_info_tasks_timeout,
            request_id=WORKER_LANDSCAPE_AGGREGATOR_REQUEST_ID,
        )
        try:
            return tasks_handler.get_worker_data_model_metadata_and_datasets()
        except (CeleryConnectionError, CeleryTaskTimeoutException) as exc:
            self._logger.warning(exc)
            return {}, {}
        except Exception:
            self._logger.error(traceback.format_exc())
            return {}, {}

    def _fetch_dmr_registries(
        self, local_workers: List[WorkerInfo]
    ) -> DataModelRegistry:
        aggregate_data_models_metadata = {}
        datasets_locations = {}
        incompatible_data_models = {}
        duplicated_datasets = {}

        for worker_info in local_workers:
            worker_id = worker_info.id
            worker_socket_addr = _get_worker_socket_addr(worker_info)
            (
                data_models_metadata,
                datasets_per_data_model,
            ) = self._get_worker_data_model_metadata_and_datasets(worker_socket_addr)

            self._process_data_models_metadata(
                data_models_metadata,
                aggregate_data_models_metadata,
                incompatible_data_models,
                worker_id,
            )
            self._process_datasets_location(
                datasets_per_data_model,
                datasets_locations,
                duplicated_datasets,
                worker_id,
            )
        self._cleanup_incompatible_data_models(
            aggregate_data_models_metadata, datasets_locations, incompatible_data_models
        )
        self._log_and_cleanup_duplicated_datasets(
            datasets_locations, duplicated_datasets
        )
        return DataModelRegistry(
            data_models_metadata=aggregate_data_models_metadata,
            datasets_locations=datasets_locations,
        )

    def _process_data_models_metadata(
        self,
        data_models_metadata: Dict[str, DataModelMetadata],
        aggregate_data_models_metadata: Dict[str, DataModelMetadata],
        incompatible_data_models: Dict[str, List[DataModelConfict]],
        worker_id: str,
    ):
        if data_models_metadata:
            for data_model, metadata in data_models_metadata.items():
                if data_model in incompatible_data_models:
                    continue
                if data_model not in aggregate_data_models_metadata:
                    aggregate_data_models_metadata[data_model] = metadata
                elif (
                    aggregate_data_models_metadata[data_model].to_dict()
                    != metadata.to_dict()
                ):
                    if data_model not in incompatible_data_models:
                        incompatible_data_models[data_model] = []
                    incompatible_data_models[data_model].append(
                        DataModelConfict(worker_id=worker_id, metadata=metadata)
                    )
                    incompatible_data_models[data_model].append(
                        DataModelConfict(
                            worker_id=worker_id,
                            metadata=aggregate_data_models_metadata[data_model],
                        )
                    )
                    aggregate_data_models_metadata.pop(data_model, None)

    def _process_datasets_location(
        self,
        datasets_per_data_model: Dict[str, List[str]],
        datasets_locations: Dict[str, Dict[str, str]],
        duplicated_datasets: Dict[str, Dict[str, List[str]]],
        worker_id: str,
    ):
        if datasets_per_data_model:
            for data_model, datasets in datasets_per_data_model.items():
                if data_model not in datasets_locations:
                    datasets_locations[data_model] = {}
                for dataset in datasets:
                    if dataset in datasets_locations[data_model]:
                        if data_model not in duplicated_datasets:
                            duplicated_datasets[data_model] = {}
                        if dataset not in duplicated_datasets[data_model]:
                            duplicated_datasets[data_model][dataset] = []
                        duplicated_datasets[data_model][dataset].append(worker_id)
                    else:
                        datasets_locations[data_model][dataset] = worker_id

    def _cleanup_incompatible_data_models(
        self,
        aggregate_data_models_metadata: Dict[str, DataModelMetadata],
        datasets_locations: Dict[str, Dict[str, str]],
        incompatible_data_models: Dict[str, List[DataModelConfict]],
    ):
        for data_model, conflicts in incompatible_data_models.items():
            workers = [conflict.worker_id for conflict in conflicts]
            metadata_1: DataModelMetadata = conflicts[0].metadata
            metadata_2: DataModelMetadata = conflicts[1].metadata
            conflicting_cdes = metadata_1.find_conflicting_fields(metadata_2)
            _log_incompatible_data_models(
                workers, data_model, conflicting_cdes, self._logger
            )
            aggregate_data_models_metadata.pop(data_model, None)
            datasets_locations.pop(data_model, None)

    def _log_and_cleanup_duplicated_datasets(
        self,
        datasets_locations: Dict[str, Dict[str, str]],
        duplicated_datasets: Dict[str, Dict[str, List[str]]],
    ):
        for data_model, datasets in duplicated_datasets.items():
            for dataset, worker_ids in datasets.items():
                _log_duplicated_dataset(worker_ids, data_model, dataset, self._logger)
                if data_model in datasets_locations:
                    datasets_locations[data_model].pop(dataset, None)
                    if not datasets_locations[data_model]:
                        del datasets_locations[data_model]


def _log_worker_changes(
    old_workers: List[WorkerInfo], new_workers: List[WorkerInfo], logger: Logger
):
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


def _log_data_model_changes(
    old_data_models: Dict[str, Dict[str, str]],
    new_data_models: Dict[str, Dict[str, str]],
    logger: Logger,
):
    added_data_models = new_data_models.keys() - old_data_models.keys()
    for data_model in added_data_models:
        log_datamodel_added(data_model, logger)
    removed_data_models = old_data_models.keys() - new_data_models.keys()
    for data_model in removed_data_models:
        log_datamodel_removed(data_model, logger)


def _log_dataset_changes(
    old_datasets_locations: Dict[str, Dict[str, str]],
    new_datasets_locations: Dict[str, Dict[str, str]],
    logger: Logger,
):
    _log_datasets_added(old_datasets_locations, new_datasets_locations, logger)
    _log_datasets_removed(old_datasets_locations, new_datasets_locations, logger)


def _log_datasets_added(
    old_datasets_locations: Dict[str, Dict[str, str]],
    new_datasets_locations: Dict[str, Dict[str, str]],
    logger: Logger,
):
    for data_model in new_datasets_locations:
        added_datasets = set(new_datasets_locations[data_model].keys())
        if data_model in old_datasets_locations:
            added_datasets -= old_datasets_locations[data_model].keys()
        for dataset in added_datasets:
            log_dataset_added(
                data_model, dataset, logger, new_datasets_locations[data_model][dataset]
            )


def _log_datasets_removed(
    old_datasets_locations: Dict[str, Dict[str, str]],
    new_datasets_locations: Dict[str, Dict[str, str]],
    logger: Logger,
):
    for data_model in old_datasets_locations:
        removed_datasets = set(old_datasets_locations[data_model].keys())
        if data_model in new_datasets_locations:
            removed_datasets -= new_datasets_locations[data_model].keys()
        for dataset in removed_datasets:
            log_dataset_removed(
                data_model, dataset, logger, old_datasets_locations[data_model][dataset]
            )


def _log_incompatible_data_models(
    workers: List[str],
    data_model: str,
    conflicting_cdes: List[str],
    logger: Logger,
):
    logger.info(
        f"Workers: '[{', '.join(workers)}]' on data model '{data_model}' have incompatibility on the following cdes: '[{', '.join(conflicting_cdes)}]'"
    )


def _log_duplicated_dataset(
    workers: List[str], data_model: str, dataset: str, logger: Logger
):
    logger.info(
        f"Dataset '{dataset}' of the data_model: '{data_model}' is not unique in the federation. Workers that contain the dataset: '[{', '.join(workers)}]'"
    )
