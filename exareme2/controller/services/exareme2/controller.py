import asyncio
import concurrent
import traceback
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from logging import Logger
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional

from exareme2 import algorithm_classes
from exareme2 import algorithm_data_loaders
from exareme2.algorithms.exareme2.algorithm import AlgorithmDataLoader
from exareme2.algorithms.exareme2.algorithm import (
    InitializationParams as AlgorithmInitParams,
)
from exareme2.algorithms.exareme2.algorithm import Variables
from exareme2.algorithms.exareme2.longitudinal_transformer import (
    DataLoader as LongitudinalTransformerRunnerDataLoader,
)
from exareme2.algorithms.exareme2.longitudinal_transformer import (
    InitializationParams as LongitudinalTransformerRunnerInitParams,
)
from exareme2.algorithms.exareme2.longitudinal_transformer import (
    LongitudinalTransformerRunner,
)
from exareme2.algorithms.specifications import TransformerName
from exareme2.controller import logger as ctrl_logger
from exareme2.controller.celery.app import CeleryConnectionError
from exareme2.controller.celery.app import CeleryTaskTimeoutException
from exareme2.controller.federation_info_logs import log_experiment_execution
from exareme2.controller.services.api.algorithm_request_dtos import AlgorithmRequestDTO
from exareme2.controller.services.exareme2.algorithm_flow_data_objects import (
    LocalWorkersTable,
)
from exareme2.controller.services.exareme2.cleaner import Cleaner
from exareme2.controller.services.exareme2.execution_engine import (
    AlgorithmExecutionEngine,
)
from exareme2.controller.services.exareme2.execution_engine import (
    AlgorithmExecutionEngineSingleLocalWorker,
)
from exareme2.controller.services.exareme2.execution_engine import CommandIdGenerator
from exareme2.controller.services.exareme2.execution_engine import (
    InitializationParams as EngineInitParams,
)
from exareme2.controller.services.exareme2.execution_engine import SMPCParams
from exareme2.controller.services.exareme2.execution_engine import Workers
from exareme2.controller.services.exareme2.tasks_handler import Exareme2TasksHandler
from exareme2.controller.services.exareme2.workers import GlobalWorker
from exareme2.controller.services.exareme2.workers import LocalWorker
from exareme2.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    WorkerLandscapeAggregator,
)
from exareme2.controller.uid_generator import UIDGenerator
from exareme2.worker_communication import InsufficientDataError
from exareme2.worker_communication import TableInfo
from exareme2.worker_communication import WorkerInfo


@dataclass(frozen=True)
class WorkersTasksHandlers:
    global_worker_tasks_handler: Optional[Exareme2TasksHandler]
    local_workers_tasks_handlers: List[Exareme2TasksHandler]


class WorkerUnresponsiveException(Exception):
    def __init__(self):
        message = (
            "One of the workers participating in the algorithm execution "
            "stopped responding"
        )
        super().__init__(message)
        self.message = message


class WorkerTaskTimeoutException(Exception):
    def __init__(self):
        message = (
            f"One of the celery in the algorithm execution took longer to finish than "
            f"the timeout.This could be caused by a high load or by an experiment with "
            f"too much data. Please try again or increase the timeout."
        )
        super().__init__(message)
        self.message = message


class DataModelViews:
    def __init__(self, local_worker_tables: Iterable[LocalWorkersTable]):
        self._views = local_worker_tables

    @classmethod
    def from_views_per_localworker(
        cls, views_per_localworker: Dict[LocalWorker, List[TableInfo]]
    ):
        return cls(
            cls._views_per_localworker_to_localworkerstables(views_per_localworker)
        )

    def to_list(self):
        return self._views

    def get_list_of_workers(self) -> List[LocalWorker]:
        """
        LocalWorkersTable is representation of a table across multiple workers. A DataModelView
        consists a collection of LocalWorkersTables that might exist on different set
        of workers or even on overlapping set of workers. This method returns a list of all
        the workers in the collection of LocalWorkersTables without duplicates
        """
        workers = set()
        for local_workers_table in self._views:
            workers.update(local_workers_table.workers_tables_info.keys())
        if not workers:
            raise InsufficientDataError(
                "None of the workers has enough data to execute the algorithm."
            )
        return list(workers)

    @classmethod
    def _views_per_localworker_to_localworkerstables(
        cls, views_per_localworker: Dict[LocalWorker, List[TableInfo]]
    ) -> List[LocalWorkersTable]:
        """
        Combines the tables of different workers into LocalWorkersTables
        """
        number_of_tables = cls._validate_number_of_views(views_per_localworker)

        local_workers_tables = [
            LocalWorkersTable(
                {worker: tables[i] for worker, tables in views_per_localworker.items()}
            )
            for i in range(number_of_tables)
        ]

        return local_workers_tables

    @classmethod
    def _validate_number_of_views(
        cls, views_per_localworker: Dict[LocalWorker, List[TableInfo]]
    ):
        """
        Checks that the number of views is the same for all workers
        """
        number_of_tables = [len(tables) for tables in views_per_localworker.values()]
        number_of_tables_equal = len(set(number_of_tables)) == 1

        if not number_of_tables_equal:
            raise ValueError(
                "The number of views is not the same for all workers"
                f" {views_per_localworker=}"
            )
        return number_of_tables[0]


class DataModelViewsCreator:
    """
    Choosing which subset of the connected to the system workers will participate in an
    execution takes place in 2 steps. The first step is choosing the workers containing
    the "data model" and datasets in the request. The second is when "data model views"
    are created. During this second step, depending on the "minimum_row_count" threshold,
    a worker that was chosen in the previous step might be left out because of
    "insufficient data". So the procedure of creating the, so called, "data model views"
    plays a key role in defining the subset of the workers that will participate in an
    execution. The DataModelCreator class implements the aforementioned functionality,
    of generating a DataModelView object as well as identifying the subset of workers
    eligible for a specific execution request.
    """

    def __init__(
        self,
        local_workers: List[LocalWorker],
        variable_groups: List[List[str]],
        var_filters: list,
        dropna: bool,
        check_min_rows: bool,
        command_id: int,
    ):
        """
        Parameters
        ----------
        local_workers: List[LocalWorker]
            The list of LocalWorkers on which the "data model views" will be created(if
            there is "sufficient data")
        variable_groups: List[List[str]]
            The variable groups
        var_filters: list
            The filtering parameters
        dropna: bool
            A boolean flag denoting if the 'Not Available' values will be kept in the
            "data model views" or not
        check_min_rows: bool
            A boolean flag denoting if a "minimum row count threshol" will be in palce
            or not
        command_id: int
            A unique id
        """
        self._local_workers = local_workers
        self._variable_groups = variable_groups
        self._var_filters = var_filters
        self._dropna = dropna
        self._check_min_rows = check_min_rows
        self._command_id = command_id

        self._data_model_views = None

    @property
    def data_model_views(self):
        return self._data_model_views

    def create_data_model_views(
        self,
    ) -> DataModelViews:
        """
        Creates the "data model views", for each variable group provided,
        using also the algorithm request arguments (data_model, datasets, filters).

        Returns
        ------
        DataModelViews
            A DataModelViews containing a view for each variable_group provided.
        """

        if self._data_model_views:
            return

        views_per_localworker = {}
        for worker in self._local_workers:
            try:
                data_model_views = worker.create_data_model_views(
                    command_id=self._command_id,
                    columns_per_view=self._variable_groups,
                    filters=self._var_filters,
                    dropna=self._dropna,
                    check_min_rows=self._check_min_rows,
                )
            except InsufficientDataError:
                continue
            views_per_localworker[worker] = data_model_views

        if not views_per_localworker:
            raise InsufficientDataError(
                "None of the workers has enough data to execute request: {LocalWorkers:"
                "Datasets}-> "
                f"{ {worker.worker_id:worker.datasets for worker in self._local_workers} } "
                f"{self._variable_groups=} {self._var_filters=} {self._dropna=} "
                f"{self._check_min_rows=}"
            )

        self._data_model_views = DataModelViews.from_views_per_localworker(
            views_per_localworker
        )


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
    also define which workers will be chosen to partiipate in a specific request for an
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

        self._workers = self._create_workers()

    def _get_workerids_for_requested_datasets(self) -> List[str]:
        return self._worker_landscape_aggregator.get_worker_ids_with_any_of_datasets(
            data_model=self._data_model, datasets=self._datasets
        )

    def _get_workerinfo_for_requested_datasets(
        self,
    ) -> List[WorkerInfo]:
        workerids = (
            self._worker_landscape_aggregator.get_worker_ids_with_any_of_datasets(
                data_model=self._data_model, datasets=self._datasets
            )
        )
        return [
            self._worker_landscape_aggregator.get_worker_info(workerid)
            for workerid in workerids
        ]

    def _get_globalworkerinfo(self) -> WorkerInfo:
        # TODO why is WorkerLandscape raising an exception when get_global_worker is called
        # and there is no Global Worker?
        try:
            return self._worker_landscape_aggregator.get_global_worker()
        except Exception:
            # means there is no global worker, single local worker execution...
            return None

    @property
    def workers(self) -> Workers:
        """
        Returns the workers (local workers and global worker) that have been selected based
        on whether they contain data belonging to the "data model" and "datasets" passed
        during the instantioation of the WorkersFederation.
        NOTE: Getting this value can be different before and after calling method
        "create_data_model_views" if the check_min_rows flag is set to True. The reason
        is that method "create_data_model_views" can potentially reject some workers,
        after applying the variables' filters and the dropna flag, on the specific
        variable groups, since some of the local workers might not contain "sufficient
        data".
        """
        return self._workers

    @property
    def worker_ids(self):
        local_worker_ids = [worker.worker_id for worker in self._workers.local_workers]

        worker_ids = []
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

    def get_global_worker_info(self) -> WorkerInfo:
        return self._worker_landscape_aggregator.get_global_worker()

    def _create_workers(self) -> Workers:
        """
        Create Workers containing only the relevant datasets
        """
        # Local Workers
        localworkersinfo = self._get_workerinfo_for_requested_datasets()
        tasks_handlers = [
            _create_tasks_handler(
                request_id=self._request_id,
                workerinfo=workerinfo,
                tasks_timeout=self._celery_tasks_timeout,
                run_udf_task_timeout=self._celery_run_udf_task_timeout,
            )
            for workerinfo in localworkersinfo
        ]

        workerids = self._get_workerids_for_requested_datasets()
        workerids_datasets = self._get_datasets_of_workerids(
            workerids, self._data_model, self._datasets
        )

        localworkers = _create_local_workers(
            request_id=self._request_id,
            context_id=self._context_id,
            data_model=self._data_model,
            workerids_datasets=workerids_datasets,
            exareme2_tasks_handlers=tasks_handlers,
        )

        # Global Worker
        globalworkerinfo = self._get_globalworkerinfo()
        if globalworkerinfo:
            tasks_handler = _create_tasks_handler(
                request_id=self._request_id,
                workerinfo=globalworkerinfo,
                tasks_timeout=self._celery_tasks_timeout,
                run_udf_task_timeout=self._celery_run_udf_task_timeout,
            )
            globalworker = _create_global_worker(
                request_id=self._request_id,
                context_id=self._context_id,
                tasks_handler=tasks_handler,
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


class ExecutionStrategy(ABC):
    """
    ExecutionStrategy is an interface, that implements a Strategy pattern, allowing to
    add arbitrary functionality before executing the final "Algorithm" logic, without
    having to alter the Controller.exec_algorithm method. Subclassing and implementing
    the abstract method run defines the desired functionality.
    """

    def __init__(
        self,
        algorithm_name: str,
        variables: Variables,
        algorithm_request_dto: AlgorithmRequestDTO,
        engine: AlgorithmExecutionEngine,
        logger: Logger,
    ):
        self._algorithm_name = algorithm_name
        self._variables = variables
        self._algorithm_data_loader = algorithm_data_loaders[algorithm_name](
            variables=variables
        )
        self._algorithm_request_dto = algorithm_request_dto
        self._engine = engine
        self._logger = logger

    @property
    def algorithm_data_loader(self):
        return self._algorithm_data_loader

    @abstractmethod
    async def run(self, data, metadata):
        pass


class LongitudinalStrategy(ExecutionStrategy):
    """
    Implementation of ExecutionStrategy interface that first executes
    "LongitudinalTransformer" and then the requested "Algorithm".
    """

    def __init__(
        self,
        algorithm_name: str,
        variables: Variables,
        algorithm_request_dto: AlgorithmRequestDTO,
        engine: AlgorithmExecutionEngine,
        logger: Logger,
    ):
        super().__init__(
            algorithm_name=algorithm_name,
            variables=variables,
            algorithm_request_dto=algorithm_request_dto,
            engine=engine,
            logger=logger,
        )

        self._algorithm_data_loader = LongitudinalTransformerRunnerDataLoader(
            variables=variables
        )

    async def run(self, data, metadata):
        init_params = LongitudinalTransformerRunnerInitParams(
            datasets=self._algorithm_request_dto.inputdata.datasets,
            var_filters=self._algorithm_request_dto.inputdata.filters,
            algorithm_parameters=self._algorithm_request_dto.preprocessing.get(
                TransformerName.LONGITUDINAL_TRANSFORMER
            ),
        )
        longitudinal_transformer = LongitudinalTransformerRunner(
            initialization_params=init_params,
            data_loader=self._algorithm_data_loader,
            engine=self._engine,
        )

        longitudinal_transform_result = await _algorithm_run_in_event_loop(
            algorithm=longitudinal_transformer,
            data_model_views=data,
            metadata=metadata,
        )
        data_transformed = longitudinal_transform_result.data
        metadata = longitudinal_transform_result.metadata

        X = data_transformed[0]
        y = data_transformed[1]
        alg_vars = Variables(x=X.columns, y=y.columns)
        algorithm_data_loader = algorithm_data_loaders[self._algorithm_name](
            variables=alg_vars
        )

        new_data_model_views = DataModelViews(data_transformed)

        algorithm_executor = AlgorithmExecutor(
            engine=self._engine,
            algorithm_data_loader=algorithm_data_loader,
            algorithm_name=self._algorithm_name,
            datasets=self._algorithm_request_dto.inputdata.datasets,
            filters=self._algorithm_request_dto.inputdata.filters,
            params=self._algorithm_request_dto.parameters,
            logger=self._logger,
        )

        algorithm_result = await algorithm_executor.run(
            data=new_data_model_views, metadata=metadata
        )
        return algorithm_result


class SingleAlgorithmStrategy(ExecutionStrategy):
    """
    Implementation of ExecutionStrategy interface that executes the requested
    "Algorithm" without any other preprocessing steps.
    """

    def __init__(
        self,
        algorithm_name: str,
        variables: Variables,
        algorithm_request_dto: AlgorithmRequestDTO,
        engine: AlgorithmExecutionEngine,
        logger: Logger,
    ):
        super().__init__(
            algorithm_name=algorithm_name,
            variables=variables,
            algorithm_request_dto=algorithm_request_dto,
            engine=engine,
            logger=logger,
        )

    async def run(self, data, metadata):
        algorithm_executor = AlgorithmExecutor(
            engine=self._engine,
            algorithm_data_loader=self._algorithm_data_loader,
            algorithm_name=self._algorithm_name,
            datasets=self._algorithm_request_dto.inputdata.datasets,
            filters=self._algorithm_request_dto.inputdata.filters,
            params=self._algorithm_request_dto.parameters,
            logger=self._logger,
        )

        algorithm_result = await algorithm_executor.run(data=data, metadata=metadata)
        return algorithm_result


class AlgorithmExecutor:
    """
    Implements the functionality of executing one "Algorithm" asynchronously
    (which must be the final step of any ExecutionStrategy)
    """

    def __init__(
        self,
        engine: AlgorithmExecutionEngine,
        algorithm_data_loader: AlgorithmDataLoader,
        algorithm_name: str,
        datasets: List[str],
        filters: dict,
        params: dict,
        logger: Logger,
    ):
        self._engine = engine

        self._algorithm_data_loader = algorithm_data_loader

        self._algorithm_name = algorithm_name
        self._datasets = datasets
        self._filters = filters
        self._params = params

        self._logger = logger

    async def run(self, data, metadata):
        # Instantiate Algorithm object
        init_params = AlgorithmInitParams(
            algorithm_name=self._algorithm_name,
            var_filters=self._filters,
            algorithm_parameters=self._params,
            datasets=self._datasets,
        )
        algorithm = algorithm_classes[self._algorithm_name](
            initialization_params=init_params,
            data_loader=self._algorithm_data_loader,
            engine=self._engine,
        )

        log_experiment_execution(
            logger=self._logger,
            request_id=self._engine._workers.local_workers[0].request_id,
            context_id=self._engine._workers.local_workers[0].context_id,
            algorithm_name=self._algorithm_name,
            datasets=self._datasets,
            algorithm_parameters=self._params,
            local_worker_ids=[
                worker.worker_id for worker in self._engine._workers.local_workers
            ],
        )

        # Call Algorithm.run inside event loop
        try:
            algorithm_result = await _algorithm_run_in_event_loop(
                algorithm=algorithm,
                data_model_views=data,
                metadata=metadata,
            )
        except CeleryConnectionError as exc:
            self._logger.error(f"ErrorType: '{type(exc)}' and message: '{exc}'")
            raise WorkerUnresponsiveException()
        except CeleryTaskTimeoutException as exc:
            self._logger.error(f"ErrorType: '{type(exc)}' and message: '{exc}'")
            raise WorkerTaskTimeoutException()
        except Exception as exc:
            self._logger.error(traceback.format_exc())
            raise exc

        return algorithm_result.json()


class Controller:
    def __init__(
        self,
        worker_landscape_aggregator: WorkerLandscapeAggregator,
        cleaner: Cleaner,
        logger: Logger,
        tasks_timeout: int,
        run_udf_task_timeout: int,
        smpc_params: SMPCParams,
    ):
        self._controller_logger = logger
        self._worker_landscape_aggregator = worker_landscape_aggregator
        self._cleaner = cleaner

        self._celery_tasks_timeout = tasks_timeout
        self._celery_run_udf_task_timeout = run_udf_task_timeout
        self._smpc_params = smpc_params

    def start_cleanup_loop(self):
        self._controller_logger.info("(Controller) Cleaner starting ...")
        self._cleaner.start()
        self._controller_logger.info("(Controller) Cleaner started.")

    def stop_cleanup_loop(self):
        self._cleaner.stop()

    async def exec_algorithm(
        self,
        algorithm_name: str,
        algorithm_request_dto: AlgorithmRequestDTO,
    ) -> str:
        command_id_generator = CommandIdGenerator()

        request_id = algorithm_request_dto.request_id
        logger = ctrl_logger.get_request_logger(request_id=request_id)
        context_id = UIDGenerator().get_a_uid()
        data_model = algorithm_request_dto.inputdata.data_model
        datasets = algorithm_request_dto.inputdata.datasets
        var_filters = algorithm_request_dto.inputdata.filters

        # Instantiate a WorkersFederation that will keep track of the local workers that are
        # relevant to the execution, based on the request parameters
        workers_federation = WorkersFederation(
            request_id=request_id,
            context_id=context_id,
            data_model=data_model,
            datasets=datasets,
            var_filters=var_filters,
            worker_landscape_aggregator=self._worker_landscape_aggregator,
            celery_tasks_timeout=self._celery_tasks_timeout,
            celery_run_udf_task_timeout=self._celery_run_udf_task_timeout,
            command_id_generator=command_id_generator,
            logger=logger,
        )

        # add the identifier of the execution(context_id), along with the relevant local
        # worker ids, to the cleaner so that whatever database artifacts are created during
        # the execution get dropped at the end of the execution, when not needed anymore
        self._cleaner.add_contextid_for_cleanup(
            context_id, workers_federation.worker_ids
        )

        # get metadata
        variable_names = (algorithm_request_dto.inputdata.x or []) + (
            algorithm_request_dto.inputdata.y or []
        )
        metadata = self._worker_landscape_aggregator.get_metadata(
            data_model=data_model, variable_names=variable_names
        )

        # instantiate an algorithm execution engine, the engine is passed to the
        # "Algorithm" implementation and serves as an API for the "Algorithm" code to
        # execute celery on workers
        engine_init_params = EngineInitParams(
            smpc_params=self._smpc_params,
            request_id=algorithm_request_dto.request_id,
            algo_flags=algorithm_request_dto.flags,
        )
        engine = _create_algorithm_execution_engine(
            engine_init_params=engine_init_params,
            command_id_generator=command_id_generator,
            workers=workers_federation.workers,
        )
        variables = Variables(
            x=sanitize_request_variable(algorithm_request_dto.inputdata.x),
            y=sanitize_request_variable(algorithm_request_dto.inputdata.y),
        )

        # Choose ExecutionStrategy
        if (
            algorithm_request_dto.preprocessing
            and algorithm_request_dto.preprocessing.get(
                TransformerName.LONGITUDINAL_TRANSFORMER
            )
        ):
            execution_strategy = LongitudinalStrategy(
                algorithm_name=algorithm_name,
                variables=variables,
                algorithm_request_dto=algorithm_request_dto,
                engine=engine,
                logger=logger,
            )
        else:
            execution_strategy = SingleAlgorithmStrategy(
                algorithm_name=algorithm_name,
                variables=variables,
                algorithm_request_dto=algorithm_request_dto,
                engine=engine,
                logger=logger,
            )

        # Create the "data model views"
        data_model_views = workers_federation.create_data_model_views(
            variable_groups=execution_strategy.algorithm_data_loader.get_variable_groups(),
            dropna=execution_strategy.algorithm_data_loader.get_dropna(),
            check_min_rows=execution_strategy.algorithm_data_loader.get_check_min_rows(),
        )

        # Execute the strategy
        algorithm_result = await execution_strategy.run(
            data=data_model_views, metadata=metadata
        )

        logger.info(f"Finished execution->  {algorithm_name=} with {request_id=}")
        logger.debug(f"Algorithm {request_id=} result-> {algorithm_result=}")

        # Cleanup artifacts created in the workers' databases during the execution
        if not self._cleaner.cleanup_context_id(context_id=context_id):
            # if the cleanup did not succeed, set the current "context_id" as released
            # so that the Cleaner retries later
            self._cleaner.release_context_id(context_id=context_id)

        return algorithm_result

    def _get_subset_of_workers_containing_datasets(self, workers, data_model, datasets):
        datasets_per_local_worker = {
            worker: self._worker_landscape_aggregator.get_worker_specific_datasets(
                worker.worker_id,
                data_model,
                datasets,
            )
            for worker in workers
        }
        return datasets_per_local_worker

    def get_datasets_locations(self) -> Dict[str, Dict[str, str]]:
        return self._worker_landscape_aggregator.get_datasets_locations()

    def get_cdes_per_data_model(self) -> dict:
        return {
            data_model: {
                column: metadata.dict() for column, metadata in cdes.values.items()
            }
            for data_model, cdes in self._worker_landscape_aggregator.get_cdes_per_data_model().items()
        }

    def get_data_models_metadata(self) -> Dict[str, Dict]:
        return {
            data_model: data_model_metadata.dict()
            for data_model, data_model_metadata in self._worker_landscape_aggregator.get_data_models_metadata().items()
        }

    def get_all_available_data_models(self) -> List[str]:
        return list(self._worker_landscape_aggregator.get_cdes_per_data_model().keys())

    def get_all_available_datasets_per_data_model(self) -> Dict[str, List[str]]:
        return (
            self._worker_landscape_aggregator.get_all_available_datasets_per_data_model()
        )


def _create_tasks_handler(
    request_id: str,
    workerinfo: WorkerInfo,
    tasks_timeout: int,
    run_udf_task_timeout: int,
):
    return Exareme2TasksHandler(
        request_id=request_id,
        worker_id=workerinfo.id,
        worker_queue_addr=str(workerinfo.ip) + ":" + str(workerinfo.port),
        worker_db_addr=str(workerinfo.db_ip) + ":" + str(workerinfo.db_port),
        tasks_timeout=tasks_timeout,
        run_udf_task_timeout=run_udf_task_timeout,
    )


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


def _create_algorithm_execution_engine(
    engine_init_params: EngineInitParams,
    command_id_generator: CommandIdGenerator,
    workers: Workers,
):
    if len(workers.local_workers) < 2:
        return AlgorithmExecutionEngineSingleLocalWorker(
            initialization_params=engine_init_params,
            command_id_generator=command_id_generator,
            workers=workers,
        )
    else:
        return AlgorithmExecutionEngine(
            initialization_params=engine_init_params,
            command_id_generator=command_id_generator,
            workers=workers,
        )


def sanitize_request_variable(variable: list):
    if variable:
        return variable
    else:
        return []


_thread_pool_executor = concurrent.futures.ThreadPoolExecutor()


# TODO add types
# TODO change func name, Transformers runs through this as well
async def _algorithm_run_in_event_loop(algorithm, data_model_views, metadata):
    # By calling blocking method Algorithm.run() inside run_in_executor(),
    # Algorithm.run() will execute in a separate thread of the threadpool and at
    # the same time yield control to the executor event loop, through await
    loop = asyncio.get_event_loop()
    algorithm_result = await loop.run_in_executor(
        _thread_pool_executor,
        algorithm.run,
        data_model_views.to_list(),
        metadata,
    )
    return algorithm_result
