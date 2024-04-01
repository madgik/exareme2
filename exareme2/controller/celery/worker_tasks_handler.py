from abc import ABC
from abc import abstractmethod
from typing import Final
from typing import List
from typing import Optional

from celery.result import AsyncResult

from exareme2.controller import logger as ctrl_logger
from exareme2.controller.celery.app import CeleryAppFactory
from exareme2.worker_communication import TableData
from exareme2.worker_communication import TableInfo
from exareme2.worker_communication import TableSchema
from exareme2.worker_communication import TableType
from exareme2.worker_communication import WorkerUDFDTO
from exareme2.worker_communication import WorkerUDFKeyArguments
from exareme2.worker_communication import WorkerUDFPosArguments
from exareme2.worker_communication import WorkerUDFResults

TASK_SIGNATURES: Final = {
    "get_tables": "exareme2.worker.exareme2.tables.tables_api.get_tables",
    "get_remote_tables": "exareme2.worker.exareme2.tables.tables_api.get_remote_tables",
    "get_merge_tables": "exareme2.worker.exareme2.tables.tables_api.get_merge_tables",
    "get_table_schema": "exareme2.worker.exareme2.tables.tables_api.get_table_schema",
    "get_table_data": "exareme2.worker.exareme2.tables.tables_api.get_table_data",
    "create_table": "exareme2.worker.exareme2.tables.tables_api.create_table",
    "create_remote_table": "exareme2.worker.exareme2.tables.tables_api.create_remote_table",
    "create_merge_table": "exareme2.worker.exareme2.tables.tables_api.create_merge_table",
    "get_views": "exareme2.worker.exareme2.views_views_api.get_views",
    "create_data_model_views": "exareme2.worker.exareme2.views.views_api.create_data_model_views",
    "run_udf": "exareme2.worker.exareme2.udfs.udfs_api.run_udf",
    "cleanup": "exareme2.worker.exareme2.cleanup.cleanup_api.cleanup",
    "validate_smpc_templates_match": "exareme2.worker.exareme2.smpc.smpc_api.validate_smpc_templates_match",
    "load_data_to_smpc_client": "exareme2.worker.exareme2.smpc.smpc_api.load_data_to_smpc_client",
    "get_smpc_result": "exareme2.worker.exareme2.smpc.smpc_api.get_smpc_result",
}


# TODO (Refactor) Split the task handlers from the celery logic
# The interface should be used in the engines and celery/grpc should implement them.
# The interface task handler should be in the services package.
class IWorkerAlgorithmTasksHandler(ABC):
    @property
    @abstractmethod
    def worker_id(self) -> str:
        pass

    @property
    @abstractmethod
    def worker_data_address(self) -> str:
        pass

    @property
    @abstractmethod
    def tasks_timeout(self) -> int:
        pass

    # TABLES functionality
    @abstractmethod
    def get_tables(self, context_id: str) -> List[str]:
        pass

    @abstractmethod
    def get_table_data(self, table_name: str) -> TableData:
        pass

    @abstractmethod
    def create_table(
        self, context_id: str, command_id: str, schema: TableSchema
    ) -> TableInfo:
        pass

    # VIEWS functionality
    @abstractmethod
    def get_views(self, context_id: str) -> List[str]:
        pass

    @abstractmethod
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
        pass

    # MERGE TABLES functionality
    @abstractmethod
    def get_merge_tables(self, context_id: str) -> List[str]:
        pass

    @abstractmethod
    def create_merge_table(
        self,
        context_id: str,
        command_id: str,
        table_infos: List[TableInfo],
    ) -> TableInfo:
        pass

    # REMOTE TABLES functionality
    @abstractmethod
    def get_remote_tables(self, context_id: str) -> List[str]:
        pass

    @abstractmethod
    def create_remote_table(
        self,
        table_name: str,
        table_schema: TableSchema,
        original_db_url: str,
    ) -> TableInfo:
        pass

    # UDFs functionality
    @abstractmethod
    def queue_run_udf(
        self,
        context_id: str,
        command_id: str,
        func_name: str,
        positional_args: WorkerUDFPosArguments,
        keyword_args: WorkerUDFKeyArguments,
        use_smpc: bool = False,
        output_schema: Optional[TableSchema] = None,
    ) -> AsyncResult:
        pass

    @abstractmethod
    def get_queued_udf_result(self, async_result: AsyncResult) -> List[WorkerUDFDTO]:
        pass

    # CLEANUP functionality
    @abstractmethod
    def queue_cleanup(self, context_id: str):
        pass

    @abstractmethod
    def wait_queued_cleanup_complete(self, async_result: AsyncResult):
        pass

    # ------------- SMPC functionality ---------------
    @abstractmethod
    def validate_smpc_templates_match(
        self,
        context_id: str,
        table_name: str,
    ):
        pass

    @abstractmethod
    def load_data_to_smpc_client(
        self, context_id: str, table_name: str, jobid: str
    ) -> str:
        pass

    @abstractmethod
    def get_smpc_result(
        self,
        jobid: str,
        context_id: str,
        command_id: str,
        command_subid: Optional[str] = "0",
    ) -> TableInfo:
        pass


class WorkerAlgorithmTasksHandler(IWorkerAlgorithmTasksHandler):
    # TODO create custom type and validator for the socket address
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

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def worker_data_address(self) -> str:
        return self._db_address

    @property
    def tasks_timeout(self) -> int:
        return self._tasks_timeout

    def _get_worker_celery_app(self):
        return CeleryAppFactory().get_celery_app(socket_addr=self._worker_queue_addr)

    # TABLES functionality
    def get_tables(self, context_id: str) -> List[str]:
        logger = ctrl_logger.get_request_logger(request_id=self._request_id)
        celery_app = self._get_worker_celery_app()
        task_signature = TASK_SIGNATURES["get_tables"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            request_id=self._request_id,
            context_id=context_id,
        )
        result = celery_app.get_result(
            async_result=async_result,
            timeout=self._tasks_timeout,
            logger=logger,
        )
        return list(result)

    def get_table_data(self, table_name: str) -> TableData:
        logger = ctrl_logger.get_request_logger(request_id=self._request_id)
        celery_app = self._get_worker_celery_app()
        task_signature = TASK_SIGNATURES["get_table_data"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            request_id=self._request_id,
            table_name=table_name,
        )
        result = celery_app.get_result(
            async_result=async_result,
            timeout=self._tasks_timeout,
            logger=logger,
        )
        return TableData.parse_raw(result)

    def create_table(
        self, context_id: str, command_id: str, schema: TableSchema
    ) -> TableInfo:
        logger = ctrl_logger.get_request_logger(request_id=self._request_id)
        schema_json = schema.json()
        celery_app = self._get_worker_celery_app()
        task_signature = TASK_SIGNATURES["create_table"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            request_id=self._request_id,
            context_id=context_id,
            command_id=command_id,
            schema_json=schema_json,
        )
        result = celery_app.get_result(
            async_result=async_result,
            timeout=self._tasks_timeout,
            logger=logger,
        )

        return TableInfo.parse_raw(result)

    # VIEWS functionality
    def get_views(self, context_id: str) -> List[str]:
        logger = ctrl_logger.get_request_logger(request_id=self._request_id)
        celery_app = self._get_worker_celery_app()
        task_signature = TASK_SIGNATURES["get_views"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            request_id=self._request_id,
            context_id=context_id,
        )
        result = celery_app.get_result(
            async_result=async_result, timeout=self._tasks_timeout, logger=logger
        )

        return result

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
        logger = ctrl_logger.get_request_logger(request_id=self._request_id)
        celery_app = self._get_worker_celery_app()
        task_signature = TASK_SIGNATURES["create_data_model_views"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            request_id=self._request_id,
            context_id=context_id,
            command_id=command_id,
            data_model=data_model,
            datasets=datasets,
            columns_per_view=columns_per_view,
            filters=filters,
            dropna=dropna,
            check_min_rows=check_min_rows,
        )
        result_str = celery_app.get_result(
            async_result=async_result, timeout=self._tasks_timeout, logger=logger
        )
        result = [TableInfo.parse_raw(res) for res in result_str]
        return result

    # MERGE TABLES functionality
    def get_merge_tables(self, context_id: str) -> List[str]:
        logger = ctrl_logger.get_request_logger(request_id=self._request_id)
        celery_app = self._get_worker_celery_app()
        task_signature = TASK_SIGNATURES["get_merge_tables"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            request_id=self._request_id,
            context_id=context_id,
        )
        result = celery_app.get_result(
            async_result=async_result,
            timeout=self._tasks_timeout,
            logger=logger,
        )

        return result

    def create_merge_table(
        self,
        context_id: str,
        command_id: str,
        table_infos: List[TableInfo],
    ) -> TableInfo:
        logger = ctrl_logger.get_request_logger(request_id=self._request_id)
        celery_app = self._get_worker_celery_app()
        task_signature = TASK_SIGNATURES["create_merge_table"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            command_id=command_id,
            request_id=self._request_id,
            context_id=context_id,
            table_infos_json=[table_info.json() for table_info in table_infos],
        )
        result = celery_app.get_result(
            async_result=async_result,
            timeout=self._tasks_timeout,
            logger=logger,
        )

        return TableInfo.parse_raw(result)

    # REMOTE TABLES functionality
    def get_remote_tables(self, context_id: str) -> List[str]:
        logger = ctrl_logger.get_request_logger(request_id=self._request_id)
        celery_app = self._get_worker_celery_app()
        task_signature = TASK_SIGNATURES["get_remote_tables"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            request_id=self._request_id,
            context_id=context_id,
        )
        result = celery_app.get_result(
            async_result=async_result,
            timeout=self._tasks_timeout,
            logger=logger,
        )
        return result

    def create_remote_table(
        self,
        table_name: str,
        table_schema: TableSchema,
        original_db_url: str,
    ) -> TableInfo:
        logger = ctrl_logger.get_request_logger(request_id=self._request_id)
        table_schema_json = table_schema.json()
        celery_app = self._get_worker_celery_app()
        task_signature = TASK_SIGNATURES["create_remote_table"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            table_name=table_name,
            table_schema_json=table_schema_json,
            monetdb_socket_address=original_db_url,
            request_id=self._request_id,
        )
        celery_app.get_result(
            async_result=async_result, timeout=self._tasks_timeout, logger=logger
        )

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
    ) -> AsyncResult:
        logger = ctrl_logger.get_request_logger(request_id=self._request_id)
        celery_app = self._get_worker_celery_app()
        task_signature = TASK_SIGNATURES["run_udf"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            command_id=command_id,
            request_id=self._request_id,
            context_id=context_id,
            func_name=func_name,
            positional_args_json=positional_args.json(),
            keyword_args_json=keyword_args.json(),
            use_smpc=use_smpc,
            output_schema=output_schema.json() if output_schema else None,
        )
        return async_result

    def get_queued_udf_result(self, async_result: AsyncResult) -> List[WorkerUDFDTO]:
        logger = ctrl_logger.get_request_logger(request_id=self._request_id)
        celery_app = self._get_worker_celery_app()
        result = celery_app.get_result(
            async_result=async_result, timeout=self._run_udf_task_timeout, logger=logger
        )
        return (WorkerUDFResults.parse_raw(result)).results

    # ------------- SMPC functionality ---------------
    def validate_smpc_templates_match(
        self,
        table_name: str,
    ):
        logger = ctrl_logger.get_request_logger(request_id=self._request_id)
        celery_app = self._get_worker_celery_app()
        task_signature = TASK_SIGNATURES["validate_smpc_templates_match"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            request_id=self._request_id,
            table_name=table_name,
        )
        celery_app.get_result(
            async_result=async_result,
            timeout=self._tasks_timeout,
            logger=logger,
        )

    def load_data_to_smpc_client(self, table_name: str, jobid: str) -> str:
        logger = ctrl_logger.get_request_logger(request_id=self._request_id)
        celery_app = self._get_worker_celery_app()
        task_signature = TASK_SIGNATURES["load_data_to_smpc_client"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            request_id=self._request_id,
            table_name=table_name,
            jobid=jobid,
        )
        result = celery_app.get_result(
            async_result=async_result, logger=logger, timeout=self._tasks_timeout
        )
        return result

    def get_smpc_result(
        self,
        jobid: str,
        context_id: str,
        command_id: str,
        command_subid: Optional[str] = "0",
    ) -> TableInfo:
        logger = ctrl_logger.get_request_logger(request_id=self._request_id)
        celery_app = self._get_worker_celery_app()
        task_signature = TASK_SIGNATURES["get_smpc_result"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            request_id=self._request_id,
            jobid=jobid,
            context_id=context_id,
            command_id=command_id,
            command_subid=command_subid,
        )
        result = celery_app.get_result(
            async_result=async_result, logger=logger, timeout=self._tasks_timeout
        )

        return TableInfo.parse_raw(result)

    # CLEANUP functionality
    def queue_cleanup(self, context_id: str):
        logger = ctrl_logger.get_request_logger(request_id=self._request_id)
        celery_app = self._get_worker_celery_app()
        task_signature = TASK_SIGNATURES["cleanup"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            request_id=self._request_id,
            context_id=context_id,
        )

        return async_result

    def wait_queued_cleanup_complete(self, async_result: AsyncResult):
        logger = ctrl_logger.get_request_logger(request_id=self._request_id)
        celery_app = self._get_worker_celery_app()
        celery_app.get_result(
            async_result=async_result,
            timeout=self._tasks_timeout,
            logger=logger,
        )
