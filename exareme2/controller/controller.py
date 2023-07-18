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
from exareme2.algorithms.algorithm import AlgorithmDataLoader
from exareme2.algorithms.algorithm import InitializationParams as AlgorithmInitParams
from exareme2.algorithms.algorithm import Variables
from exareme2.algorithms.longitudinal_transformer import (
    DataLoader as LongitudinalTransformerRunnerDataLoader,
)
from exareme2.algorithms.longitudinal_transformer import (
    InitializationParams as LongitudinalTransformerRunnerInitParams,
)
from exareme2.algorithms.longitudinal_transformer import LongitudinalTransformerRunner
from exareme2.algorithms.specifications import TransformerName
from exareme2.controller import algorithms_specifications
from exareme2.controller import controller_logger as ctrl_logger
from exareme2.controller import transformers_specifications
from exareme2.controller.algorithm_execution_engine import AlgorithmExecutionEngine
from exareme2.controller.algorithm_execution_engine import (
    AlgorithmExecutionEngineSingleLocalNode,
)
from exareme2.controller.algorithm_execution_engine import CommandIdGenerator
from exareme2.controller.algorithm_execution_engine import (
    InitializationParams as EngineInitParams,
)
from exareme2.controller.algorithm_execution_engine import Nodes
from exareme2.controller.algorithm_execution_engine_tasks_handler import (
    INodeAlgorithmTasksHandler,
)
from exareme2.controller.algorithm_execution_engine_tasks_handler import (
    NodeAlgorithmTasksHandler,
)
from exareme2.controller.algorithm_flow_data_objects import LocalNodesTable
from exareme2.controller.api.algorithm_request_dto import AlgorithmRequestDTO
from exareme2.controller.api.validator import validate_algorithm_request
from exareme2.controller.celery_app import CeleryConnectionError
from exareme2.controller.celery_app import CeleryTaskTimeoutException
from exareme2.controller.cleaner import Cleaner
from exareme2.controller.federation_info_logs import log_experiment_execution
from exareme2.controller.node_landscape_aggregator import DatasetsLocations
from exareme2.controller.node_landscape_aggregator import NodeLandscapeAggregator
from exareme2.controller.nodes import GlobalNode
from exareme2.controller.nodes import LocalNode
from exareme2.controller.uid_generator import UIDGenerator
from exareme2.exceptions import InsufficientDataError
from exareme2.node_info_DTOs import NodeInfo
from exareme2.node_tasks_DTOs import TableInfo


@dataclass(frozen=True)
class NodesTasksHandlers:
    global_node_tasks_handler: Optional[INodeAlgorithmTasksHandler]
    local_nodes_tasks_handlers: List[INodeAlgorithmTasksHandler]


@dataclass(frozen=True)
class InitializationParams:
    smpc_enabled: bool
    smpc_optional: bool
    celery_tasks_timeout: int
    celery_run_udf_task_timeout: int


class NodeUnresponsiveException(Exception):
    def __init__(self):
        message = (
            "One of the nodes participating in the algorithm execution "
            "stopped responding"
        )
        super().__init__(message)
        self.message = message


class NodeTaskTimeoutException(Exception):
    def __init__(self):
        message = (
            f"One of the tasks in the algorithm execution took longer to finish than "
            f"the timeout.This could be caused by a high load or by an experiment with "
            f"too much data. Please try again or increase the timeout."
        )
        super().__init__(message)
        self.message = message


class DataModelViews:
    def __init__(self, local_node_tables: Iterable[LocalNodesTable]):
        self._views = local_node_tables

    @classmethod
    def from_views_per_localnode(
        cls, views_per_localnode: Dict[LocalNode, List[TableInfo]]
    ):
        return cls(cls._views_per_localnode_to_localnodestables(views_per_localnode))

    def to_list(self):
        return self._views

    def get_list_of_nodes(self) -> List[LocalNode]:
        """
        LocalNodesTable is representation of a table across multiple nodes. A DataModelView
        consists a collection of LocalNodesTables that might exist on different set
        of nodes or even on overlapping set of nodes. This method returns a list of all
        the nodes in the collection of LocalNodesTables without duplicates
        """
        nodes = set()
        for local_nodes_table in self._views:
            nodes.update(local_nodes_table.nodes_tables_info.keys())
        if not nodes:
            raise InsufficientDataError(
                "None of the nodes has enough data to execute the algorithm."
            )
        return list(nodes)

    @classmethod
    def _views_per_localnode_to_localnodestables(
        cls, views_per_localnode: Dict[LocalNode, List[TableInfo]]
    ) -> List[LocalNodesTable]:
        """
        Combines the tables of different nodes into LocalNodesTables
        """
        number_of_tables = cls._validate_number_of_views(views_per_localnode)

        local_nodes_tables = [
            LocalNodesTable(
                {node: tables[i] for node, tables in views_per_localnode.items()}
            )
            for i in range(number_of_tables)
        ]

        return local_nodes_tables

    @classmethod
    def _validate_number_of_views(
        cls, views_per_localnode: Dict[LocalNode, List[TableInfo]]
    ):
        """
        Checks that the number of views is the same for all nodes
        """
        number_of_tables = [len(tables) for tables in views_per_localnode.values()]
        number_of_tables_equal = len(set(number_of_tables)) == 1

        if not number_of_tables_equal:
            raise ValueError(
                "The number of views is not the same for all nodes"
                f" {views_per_localnode=}"
            )
        return number_of_tables[0]


class DataModelViewsCreator:
    """
    Choosing which subset of the connected to the system nodes will participate in an
    execution takes place in 2 steps. The first step is choosing the nodes containing
    the "data model" and datasets in the request. The second is when "data model views"
    are created. During this second step, depending on the "minimum_row_count" threshold,
    a node that was chosen in the previous step might be left out because of
    "insufficient data". So the procedure of creating the, so called, "data model views"
    plays a key role in defining the subset of the nodes that will participate in an
    execution. The DataModelCreator class implements the aforementioned functionality,
    of generating a DataModelView object as well as identifying the subset of nodes
    eligible for a specific execution request.
    """

    def __init__(
        self,
        local_nodes: List[LocalNode],
        variable_groups: List[List[str]],
        var_filters: list,
        dropna: bool,
        check_min_rows: bool,
        command_id: int,
    ):
        """
        Parameters
        ----------
        local_nodes: List[LocalNode]
            The list of LocalNodes on which the "data model views" will be created(if
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
        self._local_nodes = local_nodes
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

        views_per_localnode = {}
        for node in self._local_nodes:
            try:
                data_model_views = node.create_data_model_views(
                    command_id=self._command_id,
                    columns_per_view=self._variable_groups,
                    filters=self._var_filters,
                    dropna=self._dropna,
                    check_min_rows=self._check_min_rows,
                )
            except InsufficientDataError:
                continue
            views_per_localnode[node] = data_model_views

        if not views_per_localnode:
            raise InsufficientDataError(
                "None of the nodes has enough data to execute request: {LocalNodes:"
                "Datasets}-> "
                f"{ {node.node_id:node.datasets for node in self._local_nodes} } "
                f"{self._variable_groups=} {self._var_filters=} {self._dropna=} "
                f"{self._check_min_rows=}"
            )

        self._data_model_views = DataModelViews.from_views_per_localnode(
            views_per_localnode
        )


class NodesFederation:
    """
    When a NodesFederation object is instantiated, with respect to the algorithm execution
    request parameters(data model, datasets), it takes care of finding which nodes of
    the federation contain the relevant data. Then, calling the create_data_model_views
    method will create the appropriate view tables in the nodes databases and return a
    DataModelViews object.

    When the system is up and running there is a number of local nodes (and one global
    node) waiting to execute tasks and return results as building blocks of executing,
    what is called in the system, an "Algorithm". The request for executing an
    "Algorithm", appart from defining which "Algorithm" to execute, contains parameters
    constraining the data on which the "Algorithm" will be executed on, like
    "data model", datasets, variables, filters etc. These constraints on the data will
    also define which nodes will be chosen to partiipate in a specific request for an
    "Algorithm" execution. The NodesFederation class implements the functionality of
    choosing (based on these request's parameters) the nodes (thus, a federation of
    nodes) that participate on a single execution request. A NodesFederation object
    is instantiated for each request to execute an "Algorithm".
    """

    def __init__(
        self,
        request_id: str,
        context_id: str,
        data_model: str,
        datasets: List[str],
        var_filters: dict,
        node_landscape_aggregator: NodeLandscapeAggregator,
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
        node_landscape_aggregator: NodeLandscapeAggregator
            The NodeLandscapeAggregator object that keeps track of the nodes currently
            connected to the system
        celery_tasks_timeout: int
            The timeout, in seconds, for the tasks to be processed by the nodes in the system
        celery_run_udf_task_timeout: int
            The timeout, in seconds, for the task executing a udf by the nodes in the system
        command_id_generator: CommandIdGenerator
            Each node task(command) is assigned a unique id, a CommandIdGenerator takes care
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

        self._node_landscape_aggregator = node_landscape_aggregator

        self._celery_tasks_timeout = celery_tasks_timeout
        self._celery_run_udf_task_timeout = celery_run_udf_task_timeout

        self._nodes = self._create_nodes()

    def _get_nodeids_for_requested_datasets(self) -> List[str]:
        return self._node_landscape_aggregator.get_node_ids_with_any_of_datasets(
            data_model=self._data_model, datasets=self._datasets
        )

    def _get_nodeinfo_for_requested_datasets(
        self,
    ) -> List[NodeInfo]:
        nodeids = self._node_landscape_aggregator.get_node_ids_with_any_of_datasets(
            data_model=self._data_model, datasets=self._datasets
        )
        return [
            self._node_landscape_aggregator.get_node_info(nodeid) for nodeid in nodeids
        ]

    def _get_globalnodeinfo(self) -> NodeInfo:
        # TODO why is NodeLandscape raising an exception when get_global_node is called
        # and there is no Global Node?
        try:
            return self._node_landscape_aggregator.get_global_node()
        except Exception:
            # means there is no global node, single local node execution...
            return None

    @property
    def nodes(self) -> Nodes:
        """
        Returns the nodes (local nodes and global node) that have been selected based
        on whether they contain data belonging to the "data model" and "datasets" passed
        during the instantioation of the NodesFederation.
        NOTE: Getting this value can be different before and after calling method
        "create_data_model_views" if the check_min_rows flag is set to True. The reason
        is that method "create_data_model_views" can potentially reject some nodes,
        after applying the variables' filters and the dropna flag, on the specific
        variable groups, since some of the local nodes might not contain "sufficient
        data".
        """
        return self._nodes

    @property
    def node_ids(self):
        local_node_ids = [node.node_id for node in self._nodes.local_nodes]

        node_ids = []
        if self._nodes.global_node:
            node_ids = [self._nodes.global_node.node_id] + local_node_ids
        else:
            node_ids = local_node_ids

        return node_ids

    def create_data_model_views(
        self, variable_groups: List[List[str]], dropna: bool, check_min_rows: bool
    ) -> DataModelViews:
        """
        Create the appropriate view tables in the nodes databases(what is called
        "data model views") and return a DataModelViews object

        Returns
        -------
        DataModelViews
            The "data model views"

        """
        data_model_views_creator = DataModelViewsCreator(
            local_nodes=self._nodes.local_nodes,
            variable_groups=variable_groups,
            var_filters=self._var_filters,
            dropna=dropna,
            check_min_rows=check_min_rows,
            command_id=self._command_id_generator.get_next_command_id(),
        )
        data_model_views_creator.create_data_model_views()

        # NOTE after creating the "data model views" some of the local nodes in the
        # original list (self._nodes.local_nodes), can be filtered out of the
        # execution if they do not contain "suffucient data", thus
        # self._nodes.local_nodes are updated
        local_nodes_filtered = (
            data_model_views_creator.data_model_views.get_list_of_nodes()
        )

        self._logger.debug(
            f"Local nodes after create_data_model_views:{local_nodes_filtered}"
        )

        # update local nodes
        self._nodes.local_nodes = local_nodes_filtered

        return data_model_views_creator.data_model_views

    def get_global_node_info(self) -> NodeInfo:
        return self._node_landscape_aggregator.get_global_node()

    def _create_nodes(self) -> Nodes:
        """
        Create Nodes containing only the relevant datasets
        """
        # Local Nodes
        localnodesinfo = self._get_nodeinfo_for_requested_datasets()
        tasks_handlers = [
            _create_node_tasks_handler(
                request_id=self._request_id,
                nodeinfo=nodeinfo,
                tasks_timeout=self._celery_tasks_timeout,
                run_udf_task_timeout=self._celery_run_udf_task_timeout,
            )
            for nodeinfo in localnodesinfo
        ]

        nodeids = self._get_nodeids_for_requested_datasets()
        nodeids_datasets = self._get_datasets_of_nodeids(
            nodeids, self._data_model, self._datasets
        )

        localnodes = _create_local_nodes(
            request_id=self._request_id,
            context_id=self._context_id,
            data_model=self._data_model,
            nodeids_datasets=nodeids_datasets,
            nodes_tasks_handlers=tasks_handlers,
        )

        # Global Node
        globalnodeinfo = self._get_globalnodeinfo()
        if globalnodeinfo:
            tasks_handler = _create_node_tasks_handler(
                request_id=self._request_id,
                nodeinfo=globalnodeinfo,
                tasks_timeout=self._celery_tasks_timeout,
                run_udf_task_timeout=self._celery_run_udf_task_timeout,
            )
            globalnode = _create_global_node(
                request_id=self._request_id,
                context_id=self._context_id,
                node_tasks_handler=tasks_handler,
            )
            nodes = Nodes(global_node=globalnode, local_nodes=localnodes)

        else:
            nodes = Nodes(local_nodes=localnodes)

        self._logger.debug(f"Created Nodes object: {nodes}")
        return nodes

    def _get_datasets_of_nodeids(
        self, nodeids: List[str], data_model: str, datasets: List[str]
    ) -> Dict[str, List[str]]:
        """
        Returns a dictionary with Node as keys and a subset or the datasets as values
        """
        datasets_per_local_node = {
            nodeid: self._node_landscape_aggregator.get_node_specific_datasets(
                nodeid,
                data_model,
                datasets,
            )
            for nodeid in nodeids
        }
        return datasets_per_local_node

    def _get_nodes_info(self) -> List[NodeInfo]:
        local_node_ids = (
            self._node_landscape_aggregator.get_node_ids_with_any_of_datasets(
                data_model=self._data_model,
                datasets=self._datasets,
            )
        )
        local_nodes_info = [
            self._node_landscape_aggregator.get_node_info(node_id)
            for node_id in local_node_ids
        ]
        return local_nodes_info


class ExecutionStrategy(ABC):
    """
    ExecutionStrategy is an interface, that implements a Strategy pattern, allowing to
    add arbitrary functionalilty before executing the final "Algorithm" logic, without
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
            request_id=self._engine._nodes.local_nodes[0].request_id,
            context_id=self._engine._nodes.local_nodes[0].context_id,
            algorithm_name=self._algorithm_name,
            datasets=self._datasets,
            algorithm_parameters=self._params,
            local_node_ids=[node.node_id for node in self._engine._nodes.local_nodes],
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
            raise NodeUnresponsiveException()
        except CeleryTaskTimeoutException as exc:
            self._logger.error(f"ErrorType: '{type(exc)}' and message: '{exc}'")
            raise NodeTaskTimeoutException()
        except Exception as exc:
            self._logger.error(traceback.format_exc())
            raise exc

        return algorithm_result.json()


class Controller:
    def __init__(
        self,
        initialization_params: InitializationParams,
        cleaner: Cleaner,
        node_landscape_aggregator: NodeLandscapeAggregator,
    ):
        self._controller_logger = ctrl_logger.get_background_service_logger()

        self._smpc_enabled = initialization_params.smpc_enabled
        self._smpc_optional = initialization_params.smpc_optional
        self._celery_tasks_timeout = initialization_params.celery_tasks_timeout
        self._celery_run_udf_task_timeout = (
            initialization_params.celery_run_udf_task_timeout
        )

        self._cleaner = cleaner
        self._node_landscape_aggregator = node_landscape_aggregator

    def start_cleanup_loop(self):
        self._controller_logger.info("(Controller) Cleaner starting ...")
        self._cleaner.start()
        self._controller_logger.info("(Controller) Cleaner started.")

    def stop_cleanup_loop(self):
        self._cleaner.stop()

    def start_node_landscape_aggregator(self):
        self._controller_logger.info(
            "(Controller) NodeLandscapeAggregator starting ..."
        )
        self._node_landscape_aggregator.start()
        self._controller_logger.info("(Controller) NodeLandscapeAggregator started.")

    def stop_node_landscape_aggregator(self):
        self._node_landscape_aggregator.stop()

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

        # Instantiate a NodesFederation that will keep track of the local nodes that are
        # relevant to the execution, based on the request parameters
        nodes_federation = NodesFederation(
            request_id=request_id,
            context_id=context_id,
            data_model=data_model,
            datasets=datasets,
            var_filters=var_filters,
            node_landscape_aggregator=self._node_landscape_aggregator,
            celery_tasks_timeout=self._celery_tasks_timeout,
            celery_run_udf_task_timeout=self._celery_run_udf_task_timeout,
            command_id_generator=command_id_generator,
            logger=logger,
        )

        # add the identifier of the execution(context_id), along with the relevant local
        # node ids, to the cleaner so that whatever database artifacts are created during
        # the execution get dropped at the end of the execution, when not needed anymore
        self._cleaner.add_contextid_for_cleanup(context_id, nodes_federation.node_ids)

        # get metadata
        variable_names = (algorithm_request_dto.inputdata.x or []) + (
            algorithm_request_dto.inputdata.y or []
        )
        metadata = self._node_landscape_aggregator.get_metadata(
            data_model=data_model, variable_names=variable_names
        )

        # instantiate an algorithm execution engine, the engine is passed to the
        # "Algorithm" implementation and serves as an API for the "Algorithm" code to
        # execute tasks on nodes
        engine_init_params = EngineInitParams(
            smpc_enabled=self._smpc_enabled,
            smpc_optional=self._smpc_optional,
            request_id=algorithm_request_dto.request_id,
            algo_flags=algorithm_request_dto.flags,
        )
        engine = _create_algorithm_execution_engine(
            engine_init_params=engine_init_params,
            command_id_generator=command_id_generator,
            nodes=nodes_federation.nodes,
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
        data_model_views = nodes_federation.create_data_model_views(
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

        # Cleanup artifacts created in the nodes' databases during the execution
        if not self._cleaner.cleanup_context_id(context_id=context_id):
            # if the cleanup did not succeed, set the current "context_id" as released
            # so that the Cleaner retries later
            self._cleaner.release_context_id(context_id=context_id)

        return algorithm_result

    def _get_subset_of_nodes_containing_datasets(self, nodes, data_model, datasets):
        datasets_per_local_node = {
            node: self._node_landscape_aggregator.get_node_specific_datasets(
                node.node_id,
                data_model,
                datasets,
            )
            for node in nodes
        }
        return datasets_per_local_node

    def validate_algorithm_execution_request(
        self, algorithm_name: str, algorithm_request_dto: AlgorithmRequestDTO
    ):
        validate_algorithm_request(
            algorithm_name=algorithm_name,
            algorithm_request_dto=algorithm_request_dto,
            algorithms_specs=algorithms_specifications,
            transformers_specs=transformers_specifications,
            node_landscape_aggregator=self._node_landscape_aggregator,
            smpc_enabled=self._smpc_enabled,
            smpc_optional=self._smpc_optional,
        )

    def get_datasets_locations(self) -> DatasetsLocations:
        return self._node_landscape_aggregator.get_datasets_locations()

    def get_cdes_per_data_model(self) -> dict:
        return {
            data_model: {
                column: metadata.dict() for column, metadata in cdes.values.items()
            }
            for data_model, cdes in self._node_landscape_aggregator.get_cdes_per_data_model().data_models_cdes.items()
        }

    def get_data_models_attributes(self) -> Dict[str, Dict]:
        return {
            data_model: data_model_metadata.dict()
            for data_model, data_model_metadata in self._node_landscape_aggregator.get_data_models_attributes().items()
        }

    def get_all_available_data_models(self) -> List[str]:
        return list(
            self._node_landscape_aggregator.get_cdes_per_data_model().data_models_cdes.keys()
        )

    def get_all_available_datasets_per_data_model(self) -> Dict[str, List[str]]:
        return (
            self._node_landscape_aggregator.get_all_available_datasets_per_data_model()
        )


def _create_node_tasks_handler(
    request_id: str, nodeinfo: NodeInfo, tasks_timeout: int, run_udf_task_timeout: int
):
    return NodeAlgorithmTasksHandler(
        request_id=request_id,
        node_id=nodeinfo.id,
        node_queue_addr=str(nodeinfo.ip) + ":" + str(nodeinfo.port),
        node_db_addr=str(nodeinfo.db_ip) + ":" + str(nodeinfo.db_port),
        tasks_timeout=tasks_timeout,
        run_udf_task_timeout=run_udf_task_timeout,
    )


def _create_local_nodes(
    request_id: str,
    context_id: str,
    data_model: str,
    nodeids_datasets: Dict[str, List[str]],
    nodes_tasks_handlers: List[INodeAlgorithmTasksHandler],
):
    local_nodes = []
    for node_tasks_handler in nodes_tasks_handlers:
        node_id = node_tasks_handler.node_id
        node = LocalNode(
            request_id=request_id,
            context_id=context_id,
            data_model=data_model,
            datasets=nodeids_datasets[node_id],
            node_tasks_handler=node_tasks_handler,
        )
        local_nodes.append(node)

    return local_nodes


def _create_global_node(request_id, context_id, node_tasks_handler):
    global_node: GlobalNode = GlobalNode(
        request_id=request_id,
        context_id=context_id,
        node_tasks_handler=node_tasks_handler,
    )
    return global_node


def _create_algorithm_execution_engine(
    engine_init_params: EngineInitParams,
    command_id_generator: CommandIdGenerator,
    nodes: Nodes,
):
    if len(nodes.local_nodes) < 2:
        return AlgorithmExecutionEngineSingleLocalNode(
            initialization_params=engine_init_params,
            command_id_generator=command_id_generator,
            nodes=nodes,
        )
    else:
        return AlgorithmExecutionEngine(
            initialization_params=engine_init_params,
            command_id_generator=command_id_generator,
            nodes=nodes,
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
