from logging import Logger
from typing import Dict
from typing import List
from typing import Optional

from exareme2.controller.services.exareme2.algorithm_flow_engine_interface import (
    CommandIdGenerator,
)
from exareme2.controller.services.exareme2.algorithm_flow_engine_interface import (
    Workers,
)
from exareme2.controller.services.exareme2.data_model_views_creator import (
    DataModelViews,
)
from exareme2.controller.services.exareme2.data_model_views_creator import (
    DataModelViewsCreator,
)
from exareme2.controller.services.exareme2.tasks_handler import Exareme2TasksHandler
from exareme2.controller.services.exareme2.workers import GlobalWorker
from exareme2.controller.services.exareme2.workers import LocalWorker
from exareme2.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    WorkerLandscapeAggregator,
)
from exareme2.worker_communication import WorkerInfo


class WorkersFederation:
    """
    When a WorkersFederation object is instantiated, with respect to the algorithm execution
    request parameters(data model, datasets), it takes care of finding which workers of
    the federation contain the relevant data. Then, calling the create_data_model_views
    method will create the appropriate view tables in the workers databases and return a
    DataModelViews object.

    When the system is up and running there is a number of local workers (and one global
    worker) waiting to execute celery and return results as building blocks of executing,
    what is called in the system, an "Algorithm". The request for executing an
    "Algorithm", appart from defining which "Algorithm" to execute, contains parameters
    constraining the data on which the "Algorithm" will be executed on, like
    "data model", datasets, variables, filters etc. These constraints on the data will
    also define which workers will be chosen to participate in a specific request for an
    "Algorithm" execution. The WorkersFederation class implements the functionality of
    choosing (based on these request's parameters) the workers (thus, a federation of
    workers) that participate on a single execution request. A WorkersFederation object
    is instantiated for each request to execute an "Algorithm".
    """

    def __init__(
        self,
        request_id: str,
        context_id: str,
        data_model: str,
        datasets: List[str],
        var_filters: dict,
        worker_landscape_aggregator: WorkerLandscapeAggregator,
        celery_tasks_timeout: int,
        celery_run_udf_task_timeout: int,
        command_id_generator: CommandIdGenerator,
        logger: Logger,
        global_worker_tasks_handler: Exareme2TasksHandler,
        local_worker_tasks_handlers: List[Exareme2TasksHandler],
    ):
        """
        Parameters
        ----------
        request_id: str
            The requests id that will uniquely idεntify the whole execution process,
            from request to result
        context_id: str
            The requests id that will uniquely idεntify the whole execution process,
            from request to result
        data_model: str
            The data model requested
        datasets: List[str]
            The datasets requested
        var_filters: dict
            The filtering parameters
        worker_landscape_aggregator: WorkerLandscapeAggregator
            The WorkerLandscapeAggregator object that keeps track of the workers currently
            connected to the system
        celery_tasks_timeout: int
            The timeout, in seconds, for the celery to be processed by the workers in the system
        celery_run_udf_task_timeout: int
            The timeout, in seconds, for the task executing a udf by the workers in the system
        command_id_generator: CommandIdGenerator
            Each worker task(command) is assigned a unique id, a CommandIdGenerator takes care
            of generating unique ids
        logger: Logger
            A logger
        """
        self._command_id_generator = command_id_generator

        self._logger = logger

        self._request_id = request_id
        self._context_id = context_id
        self._data_model = data_model
        self._datasets = datasets
        self._var_filters = var_filters
        self._worker_landscape_aggregator = worker_landscape_aggregator
        self._celery_tasks_timeout = celery_tasks_timeout
        self._celery_run_udf_task_timeout = celery_run_udf_task_timeout
        self._global_worker_tasks_handler = global_worker_tasks_handler
        self._local_worker_tasks_handlers = local_worker_tasks_handlers
        self._workers = self._create_workers()

    def _get_workerids_for_requested_datasets(self) -> List[str]:
        return self._worker_landscape_aggregator.get_worker_ids_with_any_of_datasets(
            data_model=self._data_model, datasets=self._datasets
        )

    def _get_globalworkerinfo(self) -> Optional[WorkerInfo]:
        return self._worker_landscape_aggregator.get_global_worker()

    @property
    def workers(self) -> Workers:
        return self._workers

    @property
    def worker_ids(self):
        local_worker_ids = [worker.worker_id for worker in self._workers.local_workers]

        if self._workers.global_worker:
            worker_ids = [self._workers.global_worker.worker_id] + local_worker_ids
        else:
            worker_ids = local_worker_ids

        return worker_ids

    def create_data_model_views(
        self, variable_groups: List[List[str]], dropna: bool, check_min_rows: bool
    ) -> DataModelViews:
        """
        Create the appropriate view tables in the workers databases(what is called
        "data model views") and return a DataModelViews object

        Returns
        -------
        DataModelViews
            The "data model views"

        """
        data_model_views_creator = DataModelViewsCreator(
            local_workers=self._workers.local_workers,
            variable_groups=variable_groups,
            var_filters=self._var_filters,
            dropna=dropna,
            check_min_rows=check_min_rows,
            command_id=self._command_id_generator.get_next_command_id(),
        )
        data_model_views_creator.create_data_model_views()

        # NOTE after creating the "data model views" some of the local workers in the
        # original list (self._workers.local_workers), can be filtered out of the
        # execution if they do not contain "suffucient data", thus
        # self._workers.local_workers are updated
        local_workers_filtered = (
            data_model_views_creator.data_model_views.get_list_of_workers()
        )

        self._logger.debug(
            f"Local workers after create_data_model_views:{local_workers_filtered}"
        )

        # update local workers
        self._workers.local_workers = local_workers_filtered

        return data_model_views_creator.data_model_views

    def _create_workers(self) -> Workers:
        """
        Create Workers containing only the relevant datasets
        """
        workerids = self._get_workerids_for_requested_datasets()
        workerids_datasets = self._get_datasets_of_workerids(
            workerids, self._data_model, self._datasets
        )
        localworkers = _create_local_workers(
            request_id=self._request_id,
            context_id=self._context_id,
            data_model=self._data_model,
            workerids_datasets=workerids_datasets,
            exareme2_tasks_handlers=self._local_worker_tasks_handlers,
        )

        # Global Worker
        globalworkerinfo = self._get_globalworkerinfo()
        if globalworkerinfo:
            globalworker = _create_global_worker(
                request_id=self._request_id,
                context_id=self._context_id,
                tasks_handler=self._global_worker_tasks_handler,
            )
            workers = Workers(global_worker=globalworker, local_workers=localworkers)
        else:
            workers = Workers(local_workers=localworkers)

        self._logger.debug(f"Created Workers object: {workers}")
        return workers

    def _get_datasets_of_workerids(
        self, workerids: List[str], data_model: str, datasets: List[str]
    ) -> Dict[str, List[str]]:
        """
        Returns a dictionary with Worker as keys and a subset or the datasets as values
        """
        datasets_per_local_worker = {
            workerid: self._worker_landscape_aggregator.get_worker_specific_datasets(
                workerid,
                data_model,
                datasets,
            )
            for workerid in workerids
        }
        return datasets_per_local_worker

    def _get_workers_info(self) -> List[WorkerInfo]:
        local_worker_ids = (
            self._worker_landscape_aggregator.get_worker_ids_with_any_of_datasets(
                data_model=self._data_model,
                datasets=self._datasets,
            )
        )
        local_workers_info = [
            self._worker_landscape_aggregator.get_worker_info(worker_id)
            for worker_id in local_worker_ids
        ]
        return local_workers_info


def _create_local_workers(
    request_id: str,
    context_id: str,
    data_model: str,
    workerids_datasets: Dict[str, List[str]],
    exareme2_tasks_handlers: List[Exareme2TasksHandler],
):
    local_workers = []
    for tasks_handler in exareme2_tasks_handlers:
        worker_id = tasks_handler.worker_id
        worker = LocalWorker(
            request_id=request_id,
            context_id=context_id,
            data_model=data_model,
            datasets=workerids_datasets[worker_id],
            tasks_handler=tasks_handler,
        )
        local_workers.append(worker)

    return local_workers


def _create_global_worker(request_id, context_id, tasks_handler):
    global_worker: GlobalWorker = GlobalWorker(
        request_id=request_id,
        context_id=context_id,
        tasks_handler=tasks_handler,
    )
    return global_worker
