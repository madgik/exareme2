from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

from exareme2 import DType
from exareme2.algorithms.exareme2.udfgen import make_unique_func_name
from exareme2.controller import logger as ctrl_logger
from exareme2.controller.celery.tasks_handler import WorkerTaskResult
from exareme2.controller.services.api.algorithm_request_dtos import (
    AlgorithmRequestSystemFlags,
)
from exareme2.controller.services.exareme2.algorithm_flow_data_objects import (
    AlgoFlowData,
)
from exareme2.controller.services.exareme2.algorithm_flow_data_objects import (
    GlobalWorkerData,
)
from exareme2.controller.services.exareme2.algorithm_flow_data_objects import (
    GlobalWorkerSMPCTables,
)
from exareme2.controller.services.exareme2.algorithm_flow_data_objects import (
    GlobalWorkerTable,
)
from exareme2.controller.services.exareme2.algorithm_flow_data_objects import (
    LocalWorkersData,
)
from exareme2.controller.services.exareme2.algorithm_flow_data_objects import (
    LocalWorkersSMPCTables,
)
from exareme2.controller.services.exareme2.algorithm_flow_data_objects import (
    LocalWorkersTable,
)
from exareme2.controller.services.exareme2.algorithm_flow_data_objects import (
    algoexec_udf_kwargs_to_worker_udf_kwargs,
)
from exareme2.controller.services.exareme2.algorithm_flow_data_objects import (
    algoexec_udf_posargs_to_worker_udf_posargs,
)
from exareme2.controller.services.exareme2.smpc_cluster_comm_helpers import (
    get_smpc_results,
)
from exareme2.controller.services.exareme2.smpc_cluster_comm_helpers import (
    load_data_to_smpc_clients,
)
from exareme2.controller.services.exareme2.smpc_cluster_comm_helpers import (
    trigger_smpc_operations,
)
from exareme2.controller.services.exareme2.smpc_cluster_comm_helpers import (
    wait_for_smpc_results_to_be_ready,
)
from exareme2.controller.services.exareme2.workers import GlobalWorker
from exareme2.controller.services.exareme2.workers import LocalWorker
from exareme2.smpc_cluster_communication import DifferentialPrivacyParams
from exareme2.worker_communication import SMPCTablesInfo
from exareme2.worker_communication import TableData
from exareme2.worker_communication import TableInfo
from exareme2.worker_communication import TableSchema
from exareme2.worker_communication import WorkerSMPCDTO
from exareme2.worker_communication import WorkerTableDTO
from exareme2.worker_communication import WorkerUDFDTO


@dataclass(frozen=True)
class SMPCParams:
    smpc_enabled: bool
    smpc_optional: bool
    dp_params: Optional[DifferentialPrivacyParams] = None

    def __post_init__(self):
        if self.dp_params and not self.smpc_enabled:
            raise ValueError(
                f"{self.dp_params=} but {self.smpc_enabled=}. Differential "
                "privacy mechanism needs the SMPC mechanism being enabled"
            )


@dataclass
class Workers:
    local_workers: List[LocalWorker]
    global_worker: Optional[GlobalWorker] = None


class CommandIdGenerator:
    def __init__(self):
        self._index = 0

    def get_next_command_id(self) -> str:
        current = self._index
        self._index += 1
        return str(current)


class AsyncResult:
    def get(self, timeout=None):
        pass


class InconsistentTableSchemasException(Exception):
    def __init__(self, table_infos: List[TableInfo]):
        message = f"Table_infos: {table_infos} do not have a common schema."
        super().__init__(message)


class InconsistentUDFResultSizeException(Exception):
    def __init__(self, result_tables: Dict[int, List[Tuple[LocalWorker, TableInfo]]]):
        message = (
            f"The following udf execution results on multiple workers should have "
            f"the same number of results.\nResults:{result_tables}"
        )
        super().__init__(message)


class InconsistentShareTablesValueException(Exception):
    def __init__(self, share_list: Sequence[bool], number_of_result_tables: int):
        message = f"The size of the {share_list=} does not match the {number_of_result_tables=}"
        super().__init__(message)


@dataclass(frozen=True)
class InitializationParams:
    smpc_params: SMPCParams

    request_id: str
    algo_flags: Optional[Dict[str, Any]] = None


class Exareme2AlgorithmFlowEngineInterface:
    """
    The Exareme2AlgorithmFlowEngineInterface is the class used by the algorithms to communicate with
    the workers of the system. An Exareme2AlgorithmFlowEngineInterface object is passed to all algorithms

    """

    def __init__(
        self,
        initialization_params: InitializationParams,
        command_id_generator: CommandIdGenerator,
        workers: Workers,
    ):
        self._logger = ctrl_logger.get_request_logger(
            request_id=initialization_params.request_id
        )
        self._algorithm_execution_flags = initialization_params.algo_flags
        self._smpc_params = initialization_params.smpc_params

        self._command_id_generator = command_id_generator
        self._workers = workers

    @property
    def use_smpc(self):
        return self._get_use_smpc_flag()

    @property
    def num_local_workers(self):
        # used by fed_average strategy
        return len(self._workers.local_workers)

    # UDFs functionality
    def run_udf_on_local_workers(
        self,
        func: Callable,
        positional_args: Optional[List[Any]] = None,
        keyword_args: Optional[Dict[str, Any]] = None,
        share_to_global: Union[bool, Sequence[bool]] = False,
        output_schema: Optional[List[Tuple[str, DType]]] = None,
    ) -> Union[AlgoFlowData, List[AlgoFlowData]]:
        # 1. check positional_args and keyword_args tables do not contain _GlobalWorkerTable(s)
        # 2. queues run_udf task on all local workers
        # 3. waits for all workers to complete the celery execution
        # 4. one(or multiple) new table(s) per local worker was generated
        # 5. create remote tables on global for each of the generated tables
        # 6. create merge table on global worker to merge the remote tables

        func_name = make_unique_func_name(func)
        command_id = self._command_id_generator.get_next_command_id()

        self._validate_local_run_udf_args(
            positional_args=positional_args,
            keyword_args=keyword_args,
        )

        if isinstance(share_to_global, bool):
            share_to_global = (share_to_global,)

        if output_schema:
            if len(share_to_global) != 1:
                msg = "output_schema cannot be used with multiple output UDFs."
                raise ValueError(msg)
            output_schema = TableSchema.from_list(output_schema)

        # Queue the udf on all local workers
        tasks = {}
        for worker in self._workers.local_workers:
            positional_udf_args = algoexec_udf_posargs_to_worker_udf_posargs(
                positional_args, worker
            )
            keyword_udf_args = algoexec_udf_kwargs_to_worker_udf_kwargs(
                keyword_args, worker
            )

            task = worker.queue_run_udf(
                command_id=str(command_id),
                func_name=func_name,
                positional_args=positional_udf_args,
                keyword_args=keyword_udf_args,
                use_smpc=self.use_smpc,
                output_schema=output_schema,
            )
            tasks[worker] = task

        all_workers_results = self._get_local_run_udfs_results(tasks)
        all_local_workers_data = self._convert_local_udf_results_to_local_workers_data(
            all_workers_results
        )

        # validate length of share_to_global
        number_of_results = len(all_local_workers_data)
        self._validate_share_to(share_to_global, number_of_results)

        # Share result to global worker when necessary
        results_after_sharing_step = [
            (
                self._share_local_worker_data(
                    local_workers_data, self._command_id_generator.get_next_command_id()
                )
                if share
                else local_workers_data
            )
            for share, local_workers_data in zip(
                share_to_global, all_local_workers_data
            )
        ]

        # SMPC Tables MUST be shared to the global worker
        for result in results_after_sharing_step:
            if isinstance(result, LocalWorkersSMPCTables):
                raise TypeError("SMPC should only be used when sharing the result.")

        if len(results_after_sharing_step) == 1:
            results_after_sharing_step = results_after_sharing_step[0]

        return results_after_sharing_step

    def run_udf_on_global_worker(
        self,
        func: Callable,
        positional_args: Optional[List[Any]] = None,
        keyword_args: Optional[Dict[str, Any]] = None,
        share_to_locals: Union[bool, Sequence[bool]] = False,
        output_schema: Optional[List[Tuple[str, DType]]] = None,
    ) -> Union[AlgoFlowData, List[AlgoFlowData]]:
        # 1. check positional_args and keyword_args tables do not contain _LocalWorkerTable(s)
        # 2. queue run_udf on the global worker
        # 3. wait for it to complete
        # 4. a(or multiple) new table(s) was generated on global worker
        # 5. queue create_remote_table on each of the local workers to share the generated table

        func_name = make_unique_func_name(func)
        command_id = self._command_id_generator.get_next_command_id()

        self._validate_global_run_udf_args(
            positional_args=positional_args,
            keyword_args=keyword_args,
        )

        positional_udf_args = algoexec_udf_posargs_to_worker_udf_posargs(
            positional_args
        )
        keyword_udf_args = algoexec_udf_kwargs_to_worker_udf_kwargs(keyword_args)

        if isinstance(share_to_locals, bool):
            share_to_locals = (share_to_locals,)

        if output_schema:
            if len(share_to_locals) != 1:
                msg = "output_schema cannot be used with multiple output UDFs."
                raise ValueError(msg)
            output_schema = TableSchema.from_list(output_schema)

        # Queue the udf on global worker
        task = self._workers.global_worker.queue_run_udf(
            command_id=str(command_id),
            func_name=func_name,
            positional_args=positional_udf_args,
            keyword_args=keyword_udf_args,
            use_smpc=self.use_smpc,
            output_schema=output_schema,
        )

        worker_tables = self._workers.global_worker.get_udf_result(task)
        global_worker_tables = self._convert_global_udf_results_to_global_worker_data(
            worker_tables
        )

        # validate length of share_to_locals
        number_of_results = len(global_worker_tables)
        self._validate_share_to(share_to_locals, number_of_results)

        # Share the result to local workers when necessary
        results_after_sharing_step = [
            self._share_global_table_to_locals(table) if share else table
            for share, table in zip(share_to_locals, global_worker_tables)
        ]

        if len(results_after_sharing_step) == 1:
            results_after_sharing_step = results_after_sharing_step[0]

        return results_after_sharing_step

    def _get_use_smpc_flag(self) -> bool:
        """
        SMPC usage is initially defined from the config file.

        If the smpc flag exists in the request and smpc usage is optional,
        then it's defined from the request.
        """
        flags = self._algorithm_execution_flags

        use_smpc = self._smpc_params.smpc_enabled
        if (
            self._smpc_params.smpc_optional
            and flags
            and AlgorithmRequestSystemFlags.SMPC in flags.keys()
        ):
            use_smpc = flags[AlgorithmRequestSystemFlags.SMPC]

        return use_smpc

    def _convert_global_udf_results_to_global_worker_data(
        self,
        worker_tables: List[WorkerTableDTO],
    ) -> List[GlobalWorkerTable]:
        global_tables = [
            GlobalWorkerTable(
                worker=self._workers.global_worker,
                table_info=table_dto.value,
            )
            for table_dto in worker_tables
        ]
        return global_tables

    def _share_global_table_to_locals(
        self, global_table: GlobalWorkerTable
    ) -> LocalWorkersTable:
        local_tables = {
            worker: worker.create_remote_table(
                table_name=global_table.table_info.name,
                table_schema=global_table.table_info.schema_,
                native_worker=self._workers.global_worker,
            )
            for worker in self._workers.local_workers
        }
        return LocalWorkersTable(workers_tables_info=local_tables)

    # TABLES functionality
    def get_table_data(self, worker_table) -> TableData:
        return worker_table.get_table_data()

    def get_table_schema(self, worker_table) -> TableSchema:
        return worker_table.get_table_schema()

    def _convert_local_udf_results_to_local_workers_data(
        self, all_workers_results: List[List[Tuple[LocalWorker, WorkerUDFDTO]]]
    ) -> List[LocalWorkersData]:
        results = []
        for workers_result in all_workers_results:
            # All workers' results have the same type so only the first_result is needed
            # to define the type
            first_result = workers_result[0][1]
            if isinstance(first_result, WorkerTableDTO):
                results.append(
                    LocalWorkersTable(
                        {
                            worker: worker_res.value
                            for worker, worker_res in workers_result
                        }
                    )
                )
            elif isinstance(first_result, WorkerSMPCDTO):
                results.append(
                    LocalWorkersSMPCTables(
                        {
                            worker: worker_res.value
                            for worker, worker_res in workers_result
                        }
                    )
                )
            else:
                raise NotImplementedError
        return results

    def _share_local_worker_data(
        self,
        local_workers_data: LocalWorkersData,
        command_id: int,
    ) -> GlobalWorkerData:
        if isinstance(local_workers_data, LocalWorkersTable):
            return self._share_local_table_to_global(
                local_workers_table=local_workers_data,
                command_id=command_id,
            )
        elif isinstance(local_workers_data, LocalWorkersSMPCTables):
            return self._share_local_smpc_tables_to_global(
                local_workers_data, command_id
            )

        raise NotImplementedError

    def _share_local_table_to_global(
        self,
        local_workers_table: LocalWorkersTable,
        command_id: int,
    ) -> GlobalWorkerTable:
        workers_tables = local_workers_table.workers_tables_info

        # check the tables have the same schema
        common_schema = self._validate_same_schema_tables(workers_tables)

        # create remote tables on global worker
        table_infos = [
            self._workers.global_worker.create_remote_table(
                table_name=worker_table.name,
                table_schema=common_schema,
                native_worker=worker,
            )
            for worker, worker_table in workers_tables.items()
        ]

        # merge remote tables into one merge table on global worker
        merge_table = self._workers.global_worker.create_merge_table(
            str(command_id), table_infos
        )

        return GlobalWorkerTable(
            worker=self._workers.global_worker, table_info=merge_table
        )

    def _share_local_smpc_tables_to_global(
        self,
        local_workers_smpc_tables: LocalWorkersSMPCTables,
        command_id: int,
    ) -> GlobalWorkerSMPCTables:
        global_template_table = self._share_local_table_to_global(
            local_workers_table=local_workers_smpc_tables.template_local_workers_table,
            command_id=command_id,
        )
        self._workers.global_worker.validate_smpc_templates_match(
            global_template_table.table_info.name
        )

        smpc_clients_per_op = load_data_to_smpc_clients(
            command_id, local_workers_smpc_tables
        )

        (sum_op, min_op, max_op) = trigger_smpc_operations(
            logger=self._logger,
            context_id=self._workers.global_worker.context_id,
            command_id=command_id,
            smpc_clients_per_op=smpc_clients_per_op,
            dp_params=self._smpc_params.dp_params,
        )

        wait_for_smpc_results_to_be_ready(
            logger=self._logger,
            context_id=self._workers.global_worker.context_id,
            command_id=command_id,
            sum_op=sum_op,
            min_op=min_op,
            max_op=max_op,
        )

        (
            sum_op_result_table,
            min_op_result_table,
            max_op_result_table,
        ) = get_smpc_results(
            worker=self._workers.global_worker,
            context_id=self._workers.global_worker.context_id,
            command_id=command_id,
            sum_op=sum_op,
            min_op=min_op,
            max_op=max_op,
        )

        return GlobalWorkerSMPCTables(
            worker=self._workers.global_worker,
            smpc_tables_info=SMPCTablesInfo(
                template=global_template_table.table_info,
                sum_op=sum_op_result_table,
                min_op=min_op_result_table,
                max_op=max_op_result_table,
            ),
        )

    # -------------helper methods------------
    def _validate_local_run_udf_args(
        self,
        positional_args: Optional[List[Any]] = None,
        keyword_args: Optional[Dict[str, Any]] = None,
    ):
        if self._type_exists_in_udf_args(GlobalWorkerTable):
            raise TypeError(
                f"run_udf_on_local_workers contains a 'GlobalWorkerTable' type"
                f"in the arguments which is not acceptable. "
                f"{positional_args=} \n {keyword_args=}"
            )

    def _validate_global_run_udf_args(
        self,
        positional_args: Optional[List[Any]] = None,
        keyword_args: Optional[Dict[str, Any]] = None,
    ):
        if self._type_exists_in_udf_args(LocalWorkersTable):
            raise TypeError(
                f"run_udf_on_global_worker contains a 'LocalWorkersTable' type"
                f"in the arguments which is not acceptable. "
                f"{positional_args=} \n {keyword_args=}"
            )

    def _type_exists_in_udf_args(
        self,
        input_type: type,
        positional_args: Optional[List[Any]] = None,
        keyword_args: Optional[Dict[str, Any]] = None,
    ):
        for arg in positional_args or []:
            if isinstance(arg, input_type):
                return True
        if keyword_args:
            for arg in keyword_args.values():
                if isinstance(arg, input_type):
                    return True

    def _get_local_run_udfs_results(
        self, tasks: Dict[LocalWorker, WorkerTaskResult]
    ) -> List[List[Tuple[LocalWorker, WorkerUDFDTO]]]:
        all_workers_results = {}
        for worker, task in tasks.items():
            worker_results = worker.get_udf_result(task)
            for index, worker_result in enumerate(worker_results):
                if index not in all_workers_results:
                    all_workers_results[index] = []
                all_workers_results[index].append((worker, worker_result))

        # Validate that all workers should have the same number of results from a udf
        if not all(
            len(workers_result) == len(all_workers_results[0])
            for workers_result in all_workers_results.values()
        ):
            raise InconsistentUDFResultSizeException(all_workers_results)

        # Validate that all workers have the same result type
        for workers_result in all_workers_results.values():
            if not all(
                isinstance(r, type(workers_result[0])) for r in workers_result[1:]
            ):
                raise TypeError(
                    f"The WORKERs returned results of different type. Results: {workers_result}"
                )

        all_workers_results = list(all_workers_results.values())

        return all_workers_results

    @staticmethod
    def _validate_share_to(share_to: Sequence[bool], number_of_results: int):
        if not all(isinstance(elem, bool) for elem in share_to):
            raise TypeError(
                f"share_to_locals must be of type Sequence[bool] but "
                f"{type(share_to)=} was passed"
            )
        if len(share_to) != number_of_results:
            raise InconsistentShareTablesValueException(share_to, number_of_results)

    def _validate_same_schema_tables(
        self, table_info_per_worker: Dict[LocalWorker, TableInfo]
    ) -> TableSchema:
        """
        Returns : TableSchema the common TableSchema, if all tables have the same schema
        """
        reference_schema = next(
            iter(table_info_per_worker.values())
        ).schema_  # Use the first table schema as reference
        for _, table_info in table_info_per_worker.items():
            if table_info.schema_ != reference_schema:
                raise InconsistentTableSchemasException(
                    list(table_info_per_worker.values())
                )

        return reference_schema


class Exareme2AlgorithmFlowEngineInterfaceSingleLocalWorker(
    Exareme2AlgorithmFlowEngineInterface
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._workers.global_worker = self._workers.local_workers[0]

    def _share_local_worker_data(
        self,
        local_workers_data: LocalWorkersData,
        command_id: int,
    ) -> GlobalWorkerData:
        if isinstance(local_workers_data, LocalWorkersTable):
            return GlobalWorkerTable(
                worker=self._workers.global_worker,
                table_info=local_workers_data.workers_tables_info[
                    self._workers.local_workers[0]
                ],
            )
        elif isinstance(local_workers_data, LocalWorkersSMPCTables):
            return GlobalWorkerSMPCTables(
                worker=self._workers.global_worker,
                smpc_tables_info=local_workers_data.workers_smpc_tables[
                    self._workers.global_worker
                ],
            )
        raise NotImplementedError

    def _share_global_table_to_locals(
        self, global_table: GlobalWorkerTable
    ) -> LocalWorkersTable:
        return LocalWorkersTable(
            workers_tables_info=dict(
                {self._workers.global_worker: global_table.table_info}
            )
        )
