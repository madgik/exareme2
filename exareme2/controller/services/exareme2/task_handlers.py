from typing import List
from typing import Optional

from exareme2.controller import logger as ctrl_logger
from exareme2.controller.celery.tasks_handlers import WorkerTaskResult
from exareme2.controller.celery.tasks_handlers import WorkerTasksHandler
from exareme2.worker_communication import TableData
from exareme2.worker_communication import TableInfo
from exareme2.worker_communication import TableSchema
from exareme2.worker_communication import TableType
from exareme2.worker_communication import WorkerUDFDTO
from exareme2.worker_communication import WorkerUDFKeyArguments
from exareme2.worker_communication import WorkerUDFPosArguments
from exareme2.worker_communication import WorkerUDFResults


class Exareme2TasksHandler:
    def __init__(
        self,
        request_id: str,
        worker_id: str,
        worker_queue_addr: str,
        worker_db_addr: str,
        tasks_timeout: int,
        run_udf_task_timeout: int,
    ):
        self._request_id = request_id
        self._worker_id = worker_id
        self._worker_queue_addr = worker_queue_addr
        self._db_address = worker_db_addr
        self._tasks_timeout = tasks_timeout
        self._run_udf_task_timeout = run_udf_task_timeout
        self._logger = ctrl_logger.get_request_logger(request_id=request_id)
        self._worker_tasks_handler = WorkerTasksHandler(
            self._worker_queue_addr, self._logger
        )

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def worker_data_address(self) -> str:
        return self._db_address

    @property
    def tasks_timeout(self) -> int:
        return self._tasks_timeout

    def get_tables(self, context_id: str) -> List[str]:
        result = self._worker_tasks_handler.get_tables(
            self._request_id, context_id
        ).get(self._tasks_timeout)
        return list(result)

    def get_table_data(self, table_name: str) -> TableData:
        result = self._worker_tasks_handler.get_table_data(
            request_id=self._request_id, table_name=table_name
        ).get(self._tasks_timeout)
        return TableData.parse_raw(result)

    def create_table(
        self, context_id: str, command_id: str, schema: TableSchema
    ) -> TableInfo:
        result = self._worker_tasks_handler.create_table(
            request_id=self._request_id,
            context_id=context_id,
            command_id=command_id,
            schema=schema,
        ).get(self._tasks_timeout)
        return TableInfo.parse_raw(result)

    # VIEWS functionality
    def get_views(self, context_id: str) -> List[str]:
        return self._worker_tasks_handler.get_views(
            request_id=self._request_id, context_id=context_id
        ).get(self._tasks_timeout)

    def create_data_model_views(
        self,
        context_id: str,
        command_id: str,
        data_model: str,
        datasets: List[str],
        columns_per_view: List[List[str]],
        filters: dict,
        dropna: bool = True,
        check_min_rows: bool = True,
    ) -> List[TableInfo]:
        result_str = self._worker_tasks_handler.create_data_model_views(
            request_id=self._request_id,
            context_id=context_id,
            command_id=command_id,
            data_model=data_model,
            datasets=datasets,
            columns_per_view=columns_per_view,
            filters=filters,
            dropna=dropna,
            check_min_rows=check_min_rows,
        ).get(self._tasks_timeout)
        result = [TableInfo.parse_raw(res) for res in result_str]
        return result

    # MERGE TABLES functionality
    def get_merge_tables(self, context_id: str) -> List[str]:
        return self._worker_tasks_handler.get_merge_tables(
            request_id=self._request_id,
            context_id=context_id,
        ).get(self._tasks_timeout)

    def create_merge_table(
        self,
        context_id: str,
        command_id: str,
        table_infos: List[TableInfo],
    ) -> TableInfo:
        result = self._worker_tasks_handler.create_merge_table(
            command_id=command_id,
            request_id=self._request_id,
            context_id=context_id,
            table_infos=[table_info for table_info in table_infos],
        ).get(self._tasks_timeout)

        return TableInfo.parse_raw(result)

    # REMOTE TABLES functionality
    def get_remote_tables(self, context_id: str) -> List[str]:
        result = self._worker_tasks_handler.get_remote_tables(
            request_id=self._request_id,
            context_id=context_id,
        ).get(self._tasks_timeout)
        return result

    def create_remote_table(
        self,
        table_name: str,
        table_schema: TableSchema,
        monetdb_socket_address: str,
    ) -> TableInfo:
        self._worker_tasks_handler.create_remote_table(
            request_id=self._request_id,
            table_name=table_name,
            table_schema=table_schema,
            monetdb_socket_address=monetdb_socket_address,
        ).get(self._tasks_timeout)
        return TableInfo(
            name=table_name,
            schema_=table_schema,
            type_=TableType.REMOTE,
        )

    # UDFs functionality
    def queue_run_udf(
        self,
        context_id: str,
        command_id: str,
        func_name: str,
        positional_args: WorkerUDFPosArguments,
        keyword_args: WorkerUDFKeyArguments,
        use_smpc: bool = False,
        output_schema: Optional[TableSchema] = None,
    ) -> WorkerTaskResult:
        return self._worker_tasks_handler.queue_run_udf(
            command_id=command_id,
            request_id=self._request_id,
            context_id=context_id,
            func_name=func_name,
            positional_args=positional_args,
            keyword_args=keyword_args,
            use_smpc=use_smpc,
            output_schema=output_schema,
        )

    def get_udf_result(
        self, worker_task_result: WorkerTaskResult
    ) -> List[WorkerUDFDTO]:
        result = worker_task_result.get(self._tasks_timeout)
        return (WorkerUDFResults.parse_raw(result)).results

    # ------------- SMPC functionality ---------------
    def validate_smpc_templates_match(
        self,
        table_name: str,
    ):
        self._worker_tasks_handler.validate_smpc_templates_match(
            request_id=self._request_id,
            table_name=table_name,
        ).get(self._tasks_timeout)

    def load_data_to_smpc_client(self, table_name: str, jobid: str) -> str:
        result = self._worker_tasks_handler.load_data_to_smpc_client(
            request_id=self._request_id,
            table_name=table_name,
            jobid=jobid,
        ).get(self._tasks_timeout)
        return result

    def get_smpc_result(
        self,
        jobid: str,
        context_id: str,
        command_id: str,
        command_subid: Optional[str] = "0",
    ) -> TableInfo:
        result = self._worker_tasks_handler.get_smpc_result(
            request_id=self._request_id,
            jobid=jobid,
            context_id=context_id,
            command_id=command_id,
            command_subid=command_subid,
        ).get(self._tasks_timeout)
        return TableInfo.parse_raw(result)

    def queue_cleanup(self, context_id: str):
        return self._worker_tasks_handler.queue_cleanup(
            request_id=self._request_id,
            context_id=context_id,
        )

    def wait_queued_cleanup_complete(self, worker_task_result: WorkerTaskResult):
        worker_task_result.get(self._tasks_timeout)
