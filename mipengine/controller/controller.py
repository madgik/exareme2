import asyncio
import concurrent
import traceback
from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Optional

from mipengine import algorithm_classes
from mipengine import algorithm_data_loaders
from mipengine.algorithms.algorithm import InitializationParams as AlgorithmInitParams
from mipengine.algorithms.algorithm import Variables
from mipengine.algorithms.longitudinal_transformer import (
    DataLoader as LongitudinalTransformerRunnerDataLoader,
)
from mipengine.algorithms.longitudinal_transformer import (
    InitializationParams as LongitudinalTransformerRunnerInitParams,
)
from mipengine.algorithms.longitudinal_transformer import LongitudinalTransformerRunner
from mipengine.controller import algorithms_specifications
from mipengine.controller import controller_logger as ctrl_logger
from mipengine.controller.algorithm_execution_engine import AlgorithmExecutionEngine
from mipengine.controller.algorithm_execution_engine import (
    AlgorithmExecutionEngineSingleLocalNode,
)
from mipengine.controller.algorithm_execution_engine import CommandIdGenerator
from mipengine.controller.algorithm_execution_engine import (
    InitializationParams as EngineInitParams,
)
from mipengine.controller.algorithm_execution_engine import Nodes
from mipengine.controller.algorithm_execution_engine_tasks_handler import (
    INodeAlgorithmTasksHandler,
)
from mipengine.controller.algorithm_execution_engine_tasks_handler import (
    NodeAlgorithmTasksHandler,
)
from mipengine.controller.algorithm_flow_data_objects import LocalNodesTable
from mipengine.controller.api.algorithm_request_dto import AlgorithmRequestDTO
from mipengine.controller.api.validator import validate_algorithm_request
from mipengine.controller.celery_app import CeleryConnectionError
from mipengine.controller.celery_app import CeleryTaskTimeoutException
from mipengine.controller.cleaner import Cleaner
from mipengine.controller.federation_info_logs import log_experiment_execution
from mipengine.controller.node_landscape_aggregator import DatasetsLocations
from mipengine.controller.node_landscape_aggregator import NodeLandscapeAggregator
from mipengine.controller.nodes import GlobalNode
from mipengine.controller.nodes import LocalNode
from mipengine.controller.uid_generator import UIDGenerator
from mipengine.exceptions import InsufficientDataError
from mipengine.node_info_DTOs import NodeInfo
from mipengine.node_tasks_DTOs import TableInfo


@dataclass(frozen=True)
class NodesTasksHandlers:
    global_node_tasks_handler: Optional[INodeAlgorithmTasksHandler]
    local_nodes_tasks_handlers: List[INodeAlgorithmTasksHandler]


@dataclass(frozen=True)
class _NodeInfoDTO:
    node_id: str
    queue_address: str
    db_address: str
    tasks_timeout: int
    run_udf_task_timeout: int


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
            "One of the tasks in the algorithm execution took longer to finish than the timeout."
            f"This could be caused by a high load or by an experiment with too much data. "
            f"Please try again or increase the timeout."
        )
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class DataModelViewsCreatorInitParams:
    nodes_datasets: Dict[LocalNode, List[str]]
    data_model: str
    datasets: List[str]
    variable_groups: List[List[str]]
    var_filters: list
    dropna: bool
    check_min_rows: bool
    command_id: int


class DataModelViewsCreator:
    def __init__(self, init_params: DataModelViewsCreatorInitParams):
        self._nodes_datasets = init_params.nodes_datasets
        # self._nodeids=list(self._nodes_datasets.keys())
        self._data_model = init_params.data_model
        self._datasets = init_params.datasets
        self._variable_groups = init_params.variable_groups
        self._var_filters = init_params.var_filters
        self._dropna = init_params.dropna
        self._check_min_rows = init_params.check_min_rows
        self._command_id = init_params.command_id

    def create_data_model_views(
        self,
    ) -> List[LocalNodesTable]:
        """
        Creates the data model views, for each variable group provided,
        using also the algorithm request arguments (data_model, datasets, filters).

        Returns
        ------
        List[LocalNodesTable]
        A (LocalNodesTable) view for each variable_group provided.
        """

        views_per_localnode = {}
        for node in self._nodes_datasets.keys():
            datasets = [
                datasets
                for datasets in self._datasets
                if datasets in self._nodes_datasets[node]
            ]
            try:
                data_model_views = node.create_data_model_views(
                    command_id=self._command_id,
                    data_model=self._data_model,
                    datasets=datasets,
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
                f"None of the nodes has enough data to execute request: "
                f"Nodes={[node.node_id for node in self._nodes_datasets.keys()]} "
                f"{self._data_model=} {self._datasets=} {self._variable_groups=} "
                f"{self._var_filters=} {self._dropna=} {self._check_min_rows=}"
            )

        return _data_model_views_to_localnodestables(views_per_localnode)


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

        self._thread_pool_executor = concurrent.futures.ThreadPoolExecutor()

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
        context_id = UIDGenerator().get_a_uid()
        logger = ctrl_logger.get_request_logger(
            request_id=algorithm_request_dto.request_id
        )

        data_model = algorithm_request_dto.inputdata.data_model
        datasets = algorithm_request_dto.inputdata.datasets

        if algorithm_request_dto.flags and algorithm_request_dto.flags.get(
            "longitudinal"
        ):
            longitudinal_specs = LongitudinalTransformerRunner.get_specification()
            longitudinal_algorithms = longitudinal_specs.compatible_algorithms
            if algorithm_name not in longitudinal_algorithms:
                msg = f"The algorithm provided '{algorithm_name}' is not compatible "
                msg += "with the 'longitudinal' flag."
                raise ValueError(msg)

        # instantiate nodes
        nodes = self._create_nodes(
            request_id=algorithm_request_dto.request_id,
            context_id=context_id,
            data_model=data_model,
            datasets=datasets,
        )

        # add contextid and nodeids to cleaner
        local_nodes_ids = [node.node_id for node in nodes.local_nodes]

        # global node should not be optional, it was nevertheless made optional
        # https://github.com/madgik/MIP-Engine/pull/269
        if nodes.global_node:
            all_nodes_ids = [nodes.global_node.node_id] + local_nodes_ids
        else:
            all_nodes_ids = local_nodes_ids

        self._cleaner.add_contextid_for_cleanup(context_id, all_nodes_ids)

        # get metadata
        variable_names = (algorithm_request_dto.inputdata.x or []) + (
            algorithm_request_dto.inputdata.y or []
        )
        metadata = self._node_landscape_aggregator.get_metadata(
            data_model=data_model, variable_names=variable_names
        )

        command_id_generator = CommandIdGenerator()

        variables = Variables(
            x=sanitize_request_variable(algorithm_request_dto.inputdata.x),
            y=sanitize_request_variable(algorithm_request_dto.inputdata.y),
        )

        # LONGITUDINAL specific
        if algorithm_request_dto.flags and algorithm_request_dto.flags.get(
            "longitudinal"
        ):
            data_loader = LongitudinalTransformerRunnerDataLoader(variables=variables)
        else:
            data_loader = algorithm_data_loaders[algorithm_name](variables=variables)

        nodes_datasets = self._get_subset_of_nodes_containing_datasets(
            nodes=nodes.local_nodes,
            data_model=data_model,
            datasets=datasets,
        )
        init_params = DataModelViewsCreatorInitParams(
            nodes_datasets=nodes_datasets,
            data_model=data_model,
            datasets=datasets,
            variable_groups=data_loader.get_variable_groups(),
            var_filters=algorithm_request_dto.inputdata.filters,
            dropna=data_loader.get_dropna(),
            check_min_rows=data_loader.get_check_min_rows(),
            command_id=command_id_generator.get_next_command_id(),  # should pass cmnd gen...
        )
        # create data model views
        data_model_views_creator = DataModelViewsCreator(init_params)
        data_model_views = data_model_views_creator.create_data_model_views()

        local_nodes_filtered = _get_nodes(data_model_views)

        logger.debug(f"{local_nodes_filtered=} after creating data model views")

        nodes = Nodes(global_node=nodes.global_node, local_nodes=local_nodes_filtered)

        # instantiate algorithm execution engine
        engine_init_params = EngineInitParams(
            smpc_enabled=self._smpc_enabled,
            smpc_optional=self._smpc_optional,
            request_id=algorithm_request_dto.request_id,
            context_id=context_id,
            algo_flags=algorithm_request_dto.flags,
        )
        engine = _create_algorithm_execution_engine(
            engine_init_params=engine_init_params,
            command_id_generator=command_id_generator,
            nodes=nodes,
        )

        # LONGITUDINAL specific
        if algorithm_request_dto.flags and algorithm_request_dto.flags.get(
            "longitudinal"
        ):
            longitudinal_transform_init_params = (
                LongitudinalTransformerRunnerInitParams(
                    var_filters=algorithm_request_dto.inputdata.filters,
                    algorithm_parameters=algorithm_request_dto.parameters,
                    datasets=algorithm_request_dto.inputdata.datasets,
                )
            )
            longitudinal_transformer = LongitudinalTransformerRunner(
                initialization_params=longitudinal_transform_init_params,
                data_loader=data_loader,
                engine=engine,
            )

            longitudinal_transform_result = await self._algorithm_run_in_event_loop(
                algorithm=longitudinal_transformer,
                data_model_views=data_model_views,
                metadata=metadata,
            )

            data_transformed = longitudinal_transform_result.data
            metadata = longitudinal_transform_result.metadata

            data_model_views = data_transformed

            X = data_transformed[0]
            y = data_transformed[1]
            alg_vars = Variables(x=X.columns, y=y.columns)
            data_loader = algorithm_data_loaders[algorithm_name](variables=alg_vars)

        # instantiate algorithm
        init_params = AlgorithmInitParams(
            algorithm_name=algorithm_name,
            var_filters=algorithm_request_dto.inputdata.filters,
            algorithm_parameters=algorithm_request_dto.parameters,
            datasets=algorithm_request_dto.inputdata.datasets,
        )
        algorithm = algorithm_classes[algorithm_name](
            initialization_params=init_params,
            data_loader=data_loader,
            engine=engine,
        )

        log_experiment_execution(
            logger=logger,
            request_id=algorithm_request_dto.request_id,
            context_id=context_id,
            algorithm_name=algorithm_name,
            datasets=algorithm_request_dto.inputdata.datasets,
            algorithm_parameters=algorithm_request_dto.json(),
            local_node_ids=local_nodes_ids,
        )

        # run the algorithm
        try:
            algorithm_result = await self._algorithm_run_in_event_loop(
                algorithm=algorithm,
                data_model_views=data_model_views,
                metadata=metadata,
            )
        except CeleryConnectionError as exc:
            logger.error(f"ErrorType: '{type(exc)}' and message: '{exc}'")
            raise NodeUnresponsiveException()
        except CeleryTaskTimeoutException as exc:
            logger.error(f"ErrorType: '{type(exc)}' and message: '{exc}'")
            raise NodeTaskTimeoutException()
        except Exception as exc:
            logger.error(traceback.format_exc())
            raise exc

        finally:
            if not self._cleaner.cleanup_context_id(context_id=context_id):
                self._cleaner.release_context_id(context_id=context_id)

        logger.info(
            f"Finished execution->  {algorithm_name=} with {algorithm_request_dto.request_id=}"
        )
        logger.debug(
            f"Algorithm {algorithm_request_dto.request_id=} result-> {algorithm_result.json()=}"
        )
        return algorithm_result.json()

    # TODO add types
    async def _algorithm_run_in_event_loop(self, algorithm, data_model_views, metadata):
        # By calling blocking method Algorithm.run() inside run_in_executor(),
        # Algorithm.run() will execute in a separate thread of the threadpool and at
        # the same time yield control to the executor event loop, through await
        loop = asyncio.get_event_loop()
        algorithm_result = await loop.run_in_executor(
            self._thread_pool_executor,
            algorithm.run,
            data_model_views,
            metadata,
        )
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
        available_datasets_per_data_model = (
            self.get_all_available_datasets_per_data_model()
        )
        validate_algorithm_request(
            algorithm_name=algorithm_name,
            algorithm_request_dto=algorithm_request_dto,
            available_datasets_per_data_model=available_datasets_per_data_model,
            algorithms_specs=algorithms_specifications,
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

    def get_all_local_nodes(self) -> List[NodeInfo]:
        return self._node_landscape_aggregator.get_all_local_nodes()

    def get_global_node(self) -> NodeInfo:
        return self._node_landscape_aggregator.get_global_node()

    def _get_node_info_by_id(self, node_id: str) -> _NodeInfoDTO:
        node = self._node_landscape_aggregator.get_node_info(node_id)

        return _NodeInfoDTO(
            node_id=node.id,
            queue_address=":".join([str(node.ip), str(node.port)]),
            db_address=":".join([str(node.db_ip), str(node.db_port)]),
            tasks_timeout=self._celery_tasks_timeout,
            run_udf_task_timeout=self._celery_run_udf_task_timeout,
        )

    def _get_nodes_info_by_dataset(
        self, data_model: str, datasets: List[str]
    ) -> List[_NodeInfoDTO]:
        local_node_ids = (
            self._node_landscape_aggregator.get_node_ids_with_any_of_datasets(
                data_model=data_model,
                datasets=datasets,
            )
        )
        local_nodes_info = [
            self._node_landscape_aggregator.get_node_info(node_id)
            for node_id in local_node_ids
        ]
        nodes_info = []
        for local_node in local_nodes_info:
            nodes_info.append(
                _NodeInfoDTO(
                    node_id=local_node.id,
                    queue_address=":".join([str(local_node.ip), str(local_node.port)]),
                    db_address=":".join(
                        [str(local_node.db_ip), str(local_node.db_port)]
                    ),
                    tasks_timeout=self._celery_tasks_timeout,
                    run_udf_task_timeout=self._celery_run_udf_task_timeout,
                )
            )

        return nodes_info

    def _create_nodes_tasks_handlers(
        self, data_model: str, datasets: List[str]
    ) -> NodesTasksHandlers:
        # Get only the relevant nodes
        local_nodes_info = self._get_nodes_info_by_dataset(
            data_model=data_model, datasets=datasets
        )
        local_nodes_tasks_handlers = [
            NodeAlgorithmTasksHandler(
                node_id=node_info.node_id,
                node_queue_addr=node_info.queue_address,
                node_db_addr=node_info.db_address,
                tasks_timeout=node_info.tasks_timeout,
                run_udf_task_timeout=node_info.run_udf_task_timeout,
            )
            for node_info in local_nodes_info
        ]

        global_node_tasks_handler = None
        try:
            # raises exception if there is no global node...
            global_node = self._node_landscape_aggregator.get_global_node()

            global_node_tasks_handler = NodeAlgorithmTasksHandler(
                node_id=global_node.id,
                node_queue_addr=":".join([str(global_node.ip), str(global_node.port)]),
                node_db_addr=":".join(
                    [str(global_node.db_ip), str(global_node.db_port)]
                ),
                tasks_timeout=self._celery_tasks_timeout,
                run_udf_task_timeout=self._celery_run_udf_task_timeout,
            )

            # global node should not be optional, it was nevertheless made optional
            # https://github.com/madgik/MIP-Engine/pull/269
        except Exception:
            pass

        return NodesTasksHandlers(
            global_node_tasks_handler=global_node_tasks_handler,
            local_nodes_tasks_handlers=local_nodes_tasks_handlers,
        )

    def _create_nodes(self, request_id, context_id, data_model, datasets):
        node_tasks_handlers = self._create_nodes_tasks_handlers(
            data_model=data_model, datasets=datasets
        )

        # global node should not be optional, it was nevertheless made optional
        # https://github.com/madgik/MIP-Engine/pull/269
        global_node_tasks_handler = node_tasks_handlers.global_node_tasks_handler
        global_node = (
            _create_global_node(request_id, context_id, global_node_tasks_handler)
            if global_node_tasks_handler
            else None
        )

        nodes = Nodes(
            global_node=global_node,
            local_nodes=_create_local_nodes(
                request_id, context_id, node_tasks_handlers.local_nodes_tasks_handlers
            ),
        )

        return nodes


def _get_nodes(local_nodes_tables: List[LocalNodesTable]) -> List[LocalNode]:
    valid_nodes = set()
    for local_node_table in local_nodes_tables:
        valid_nodes.update(local_node_table.nodes_tables_info.keys())
    if not valid_nodes:
        raise InsufficientDataError(
            "None of the nodes has enough data to execute the algorithm."
        )
    return list(valid_nodes)


def _create_local_nodes(request_id, context_id, nodes_tasks_handlers):
    local_nodes: List[LocalNode] = [
        LocalNode(
            request_id=request_id,
            context_id=context_id,
            node_tasks_handler=node_tasks_handler,
        )
        for node_tasks_handler in nodes_tasks_handlers
    ]
    return local_nodes


def _create_global_node(request_id, context_id, node_tasks_handler):
    global_node: GlobalNode = GlobalNode(
        request_id=request_id,
        context_id=context_id,
        node_tasks_handler=node_tasks_handler,
    )
    return global_node


def _data_model_views_to_localnodestables(
    views_per_localnode: Dict[LocalNode, List[TableInfo]]
) -> List[LocalNodesTable]:

    number_of_tables = _validate_number_of_views(views_per_localnode)

    local_nodes_tables = [
        LocalNodesTable(
            {node: tables[i] for node, tables in views_per_localnode.items()}
        )
        for i in range(number_of_tables)
    ]

    return local_nodes_tables


def _validate_number_of_views(views_per_localnode: dict):
    number_of_tables = [len(tables) for tables in views_per_localnode.values()]
    number_of_tables_equal = len(set(number_of_tables)) == 1

    if not number_of_tables_equal:
        raise ValueError(
            f"The number of views does is not equal for all nodes {views_per_localnode=}"
        )
    return number_of_tables[0]


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
