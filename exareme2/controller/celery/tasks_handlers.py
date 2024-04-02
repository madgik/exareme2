from logging import Logger
from typing import Dict
from typing import Final
from typing import List
from typing import Optional

from celery.result import AsyncResult

from exareme2.celery_app_conf import CELERY_APP_QUEUE_MAX_PRIORITY
from exareme2.controller import logger as ctrl_logger
from exareme2.controller.celery.app import CeleryAppFactory
from exareme2.controller.celery.app import CeleryWrapper
from exareme2.worker_communication import CommonDataElements
from exareme2.worker_communication import DataModelAttributes
from exareme2.worker_communication import TableData
from exareme2.worker_communication import TableInfo
from exareme2.worker_communication import TableSchema
from exareme2.worker_communication import TableType
from exareme2.worker_communication import WorkerInfo
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
    "get_views": "exareme2.worker.exareme2.views.views_api.get_views",
    "create_data_model_views": "exareme2.worker.exareme2.views.views_api.create_data_model_views",
    "run_udf": "exareme2.worker.exareme2.udfs.udfs_api.run_udf",
    "cleanup": "exareme2.worker.exareme2.cleanup.cleanup_api.cleanup",
    "validate_smpc_templates_match": "exareme2.worker.exareme2.smpc.smpc_api.validate_smpc_templates_match",
    "load_data_to_smpc_client": "exareme2.worker.exareme2.smpc.smpc_api.load_data_to_smpc_client",
    "get_smpc_result": "exareme2.worker.exareme2.smpc.smpc_api.get_smpc_result",
    "get_worker_info": "exareme2.worker.worker_info.worker_info_api.get_worker_info",
    "get_worker_datasets_per_data_model": "exareme2.worker.worker_info.worker_info_api.get_worker_datasets_per_data_model",
    "get_data_model_cdes": "exareme2.worker.worker_info.worker_info_api.get_data_model_cdes",
    "get_data_model_attributes": "exareme2.worker.worker_info.worker_info_api.get_data_model_attributes",
    "healthcheck": "exareme2.worker.worker_info.worker_info_api.healthcheck",
}


class WorkerTaskResult:
    def __init__(
        self, celery_app: CeleryWrapper, async_result: AsyncResult, logger: Logger
    ):
        self._celery_app = celery_app
        self._async_result = async_result
        self._logger = logger

    def get(self, timeout: int):
        return self._celery_app.get_result(
            async_result=self._async_result, timeout=timeout, logger=self._logger
        )


class WorkerTasksHandler:
    def __init__(self, worker_queue_addr: str, logger: Logger):
        self._worker_queue_addr = worker_queue_addr
        self._logger = logger

    def _get_worker_celery_app(self):
        return CeleryAppFactory().get_celery_app(socket_addr=self._worker_queue_addr)
        # TABLES functionality

    def get_tables(self, request_id: str, context_id: str) -> WorkerTaskResult:
        celery_app = self._get_worker_celery_app()
        task_signature = TASK_SIGNATURES["get_tables"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=self._logger,
            request_id=request_id,
            context_id=context_id,
        )

        return WorkerTaskResult(celery_app, async_result, self._logger)

    def get_table_data(self, request_id: str, table_name: str) -> WorkerTaskResult:
        logger = ctrl_logger.get_request_logger(request_id=request_id)
        celery_app = self._get_worker_celery_app()
        task_signature = TASK_SIGNATURES["get_table_data"]

        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            request_id=request_id,
            table_name=table_name,
        )
        return WorkerTaskResult(celery_app, async_result, self._logger)

    def create_table(
        self, request_id: str, context_id: str, command_id: str, schema: TableSchema
    ) -> WorkerTaskResult:
        logger = ctrl_logger.get_request_logger(request_id=request_id)
        schema_json = schema.json()
        celery_app = self._get_worker_celery_app()
        task_signature = TASK_SIGNATURES["create_table"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            request_id=request_id,
            context_id=context_id,
            command_id=command_id,
            schema_json=schema_json,
        )
        return WorkerTaskResult(celery_app, async_result, self._logger)

    # VIEWS functionality
    def get_views(self, request_id: str, context_id: str) -> WorkerTaskResult:
        logger = ctrl_logger.get_request_logger(request_id=request_id)
        celery_app = self._get_worker_celery_app()
        task_signature = TASK_SIGNATURES["get_views"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            request_id=request_id,
            context_id=context_id,
        )
        return WorkerTaskResult(celery_app, async_result, self._logger)

    def create_data_model_views(
        self,
        request_id: str,
        context_id: str,
        command_id: str,
        data_model: str,
        datasets: List[str],
        columns_per_view: List[List[str]],
        filters: dict,
        dropna: bool = True,
        check_min_rows: bool = True,
    ) -> WorkerTaskResult:
        logger = ctrl_logger.get_request_logger(request_id=request_id)
        celery_app = self._get_worker_celery_app()
        task_signature = TASK_SIGNATURES["create_data_model_views"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            request_id=request_id,
            context_id=context_id,
            command_id=command_id,
            data_model=data_model,
            datasets=datasets,
            columns_per_view=columns_per_view,
            filters=filters,
            dropna=dropna,
            check_min_rows=check_min_rows,
        )
        return WorkerTaskResult(celery_app, async_result, self._logger)

    # MERGE TABLES functionality
    def get_merge_tables(self, request_id: str, context_id: str) -> WorkerTaskResult:
        logger = ctrl_logger.get_request_logger(request_id=request_id)
        celery_app = self._get_worker_celery_app()
        task_signature = TASK_SIGNATURES["get_merge_tables"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            request_id=request_id,
            context_id=context_id,
        )
        return WorkerTaskResult(celery_app, async_result, self._logger)

    def create_merge_table(
        self,
        request_id: str,
        context_id: str,
        command_id: str,
        table_infos: List[TableInfo],
    ) -> WorkerTaskResult:
        logger = ctrl_logger.get_request_logger(request_id=request_id)
        celery_app = self._get_worker_celery_app()
        task_signature = TASK_SIGNATURES["create_merge_table"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            command_id=command_id,
            request_id=request_id,
            context_id=context_id,
            table_infos_json=[table_info.json() for table_info in table_infos],
        )
        return WorkerTaskResult(celery_app, async_result, self._logger)

    # REMOTE TABLES functionality
    def get_remote_tables(self, request_id: str, context_id: str) -> WorkerTaskResult:
        logger = ctrl_logger.get_request_logger(request_id=request_id)
        celery_app = self._get_worker_celery_app()
        task_signature = TASK_SIGNATURES["get_remote_tables"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            request_id=request_id,
            context_id=context_id,
        )
        return WorkerTaskResult(celery_app, async_result, self._logger)

    def create_remote_table(
        self,
        request_id: str,
        table_name: str,
        table_schema: TableSchema,
        monetdb_socket_address: str,
    ) -> WorkerTaskResult:
        logger = ctrl_logger.get_request_logger(request_id=request_id)
        table_schema_json = table_schema.json()
        celery_app = self._get_worker_celery_app()
        task_signature = TASK_SIGNATURES["create_remote_table"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            table_name=table_name,
            table_schema_json=table_schema_json,
            monetdb_socket_address=monetdb_socket_address,
            request_id=request_id,
        )
        return WorkerTaskResult(celery_app, async_result, self._logger)

    # UDFs functionality
    def queue_run_udf(
        self,
        request_id: str,
        context_id: str,
        command_id: str,
        func_name: str,
        positional_args: WorkerUDFPosArguments,
        keyword_args: WorkerUDFKeyArguments,
        use_smpc: bool = False,
        output_schema: Optional[TableSchema] = None,
    ) -> WorkerTaskResult:
        logger = ctrl_logger.get_request_logger(request_id=request_id)
        celery_app = self._get_worker_celery_app()
        task_signature = TASK_SIGNATURES["run_udf"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            command_id=command_id,
            request_id=request_id,
            context_id=context_id,
            func_name=func_name,
            positional_args_json=positional_args.json(),
            keyword_args_json=keyword_args.json(),
            use_smpc=use_smpc,
            output_schema=output_schema.json() if output_schema else None,
        )
        return WorkerTaskResult(celery_app, async_result, self._logger)

    # ------------- SMPC functionality ---------------
    def validate_smpc_templates_match(
        self,
        request_id: str,
        table_name: str,
    ):
        logger = ctrl_logger.get_request_logger(request_id=request_id)
        celery_app = self._get_worker_celery_app()
        task_signature = TASK_SIGNATURES["validate_smpc_templates_match"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            request_id=request_id,
            table_name=table_name,
        )
        return WorkerTaskResult(celery_app, async_result, self._logger)

    def load_data_to_smpc_client(
        self, request_id: str, table_name: str, jobid: str
    ) -> WorkerTaskResult:
        logger = ctrl_logger.get_request_logger(request_id=request_id)
        celery_app = self._get_worker_celery_app()
        task_signature = TASK_SIGNATURES["load_data_to_smpc_client"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            request_id=request_id,
            table_name=table_name,
            jobid=jobid,
        )
        return WorkerTaskResult(celery_app, async_result, self._logger)

    def get_smpc_result(
        self,
        request_id: str,
        jobid: str,
        context_id: str,
        command_id: str,
        command_subid: Optional[str] = "0",
    ) -> WorkerTaskResult:
        logger = ctrl_logger.get_request_logger(request_id=request_id)
        celery_app = self._get_worker_celery_app()
        task_signature = TASK_SIGNATURES["get_smpc_result"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            request_id=request_id,
            jobid=jobid,
            context_id=context_id,
            command_id=command_id,
            command_subid=command_subid,
        )
        return WorkerTaskResult(celery_app, async_result, self._logger)

    # CLEANUP functionality
    def queue_cleanup(self, request_id: str, context_id: str):
        logger = ctrl_logger.get_request_logger(request_id=request_id)
        celery_app = self._get_worker_celery_app()
        task_signature = TASK_SIGNATURES["cleanup"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            request_id=request_id,
            context_id=context_id,
        )
        return WorkerTaskResult(celery_app, async_result, self._logger)

    def queue_worker_info_task(self, request_id: str) -> WorkerTaskResult:
        celery_app = self._get_worker_celery_app()
        task_signature = TASK_SIGNATURES["get_worker_info"]
        logger = ctrl_logger.get_request_logger(request_id=request_id)
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            request_id=request_id,
            priority=CELERY_APP_QUEUE_MAX_PRIORITY,
        )
        return WorkerTaskResult(celery_app, async_result, self._logger)

    # --------------- get_worker_datasets_per_data_model task ---------------
    # NON-BLOCKING
    def queue_worker_datasets_per_data_model_task(
        self, request_id: str
    ) -> WorkerTaskResult:
        celery_app = self._get_worker_celery_app()
        task_signature = TASK_SIGNATURES["get_worker_datasets_per_data_model"]
        logger = ctrl_logger.get_request_logger(request_id=request_id)
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            request_id=request_id,
            priority=CELERY_APP_QUEUE_MAX_PRIORITY,
        )
        return WorkerTaskResult(celery_app, async_result, self._logger)

    # --------------- get_data_model_cdes task ---------------
    # NON-BLOCKING
    def queue_data_model_cdes_task(
        self, request_id: str, data_model: str
    ) -> WorkerTaskResult:
        celery_app = self._get_worker_celery_app()
        task_signature = TASK_SIGNATURES["get_data_model_cdes"]
        logger = ctrl_logger.get_request_logger(request_id=request_id)
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            request_id=request_id,
            data_model=data_model,
            priority=CELERY_APP_QUEUE_MAX_PRIORITY,
        )
        return WorkerTaskResult(celery_app, async_result, self._logger)

    # --------------- get_data_model_attributes task ---------------
    def queue_data_model_attributes_task(
        self, request_id: str, data_model: str
    ) -> WorkerTaskResult:
        celery_app = self._get_worker_celery_app()
        task_signature = TASK_SIGNATURES["get_data_model_attributes"]
        logger = ctrl_logger.get_request_logger(request_id=request_id)
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            request_id=request_id,
            data_model=data_model,
            priority=CELERY_APP_QUEUE_MAX_PRIORITY,
        )
        return WorkerTaskResult(celery_app, async_result, self._logger)

    # --------------- healthcheck task ---------------
    # NON-BLOCKING
    def queue_healthcheck_task(
        self, request_id: str, check_db: bool
    ) -> WorkerTaskResult:
        celery_app = self._get_worker_celery_app()
        task_signature = TASK_SIGNATURES["healthcheck"]
        logger = ctrl_logger.get_request_logger(request_id=request_id)
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            request_id=request_id,
            check_db=check_db,
            priority=CELERY_APP_QUEUE_MAX_PRIORITY,
        )
        return WorkerTaskResult(celery_app, async_result, self._logger)


class WorkerInfoTasksHandler:
    def __init__(self, worker_queue_addr: str, tasks_timeout: int, request_id: str):
        self._worker_queue_addr = worker_queue_addr
        self._tasks_timeout = tasks_timeout
        self._request_id = request_id
        self._logger = ctrl_logger.get_request_logger(request_id=request_id)
        self._worker_tasks_handler = WorkerTasksHandler(
            self._worker_queue_addr, self._logger
        )

    def get_worker_info_task(self) -> WorkerInfo:
        result = self._worker_tasks_handler.queue_worker_info_task(
            self._request_id
        ).get(self._tasks_timeout)
        return WorkerInfo.parse_raw(result)

    def get_worker_datasets_per_data_model_task(self) -> Dict[str, Dict[str, str]]:
        return self._worker_tasks_handler.queue_worker_datasets_per_data_model_task(
            self._request_id
        ).get(self._tasks_timeout)

    def get_data_model_cdes_task(self, data_model: str) -> CommonDataElements:
        result = self._worker_tasks_handler.queue_data_model_cdes_task(
            request_id=self._request_id,
            data_model=data_model,
        ).get(self._tasks_timeout)
        return CommonDataElements.parse_raw(result)

    def get_data_model_attributes_task(self, data_model: str) -> DataModelAttributes:
        result = self._worker_tasks_handler.queue_data_model_attributes_task(
            self._request_id, data_model
        ).get(self._tasks_timeout)
        return DataModelAttributes.parse_raw(result)

    def get_healthcheck_task(self, check_db: bool):
        return self._worker_tasks_handler.queue_healthcheck_task(
            request_id=self._request_id,
            check_db=check_db,
        ).get(self._tasks_timeout)


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

    # TABLES functionality
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

    def get_queued_udf_result(
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

    # CLEANUP functionality
    def queue_cleanup(self, context_id: str):
        return self._worker_tasks_handler.queue_cleanup(
            request_id=self._request_id,
            context_id=context_id,
        )

    def wait_queued_cleanup_complete(self, worker_task_result: WorkerTaskResult):
        worker_task_result.get(self._tasks_timeout)
