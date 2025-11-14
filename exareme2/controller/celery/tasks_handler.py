from logging import Logger
from typing import Final
from typing import List
from typing import Optional

from celery.result import AsyncResult

from exareme2.celery_app_conf import CELERY_APP_QUEUE_MAX_PRIORITY
from exareme2.controller.celery.app import CeleryAppFactory
from exareme2.controller.celery.app import CeleryWrapper
from exareme2.worker_communication import TableInfo
from exareme2.worker_communication import TableSchema
from exareme2.worker_communication import WorkerUDFKeyArguments
from exareme2.worker_communication import WorkerUDFPosArguments

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
    "start_flower_client": "exareme2.worker.flower.starter.starter_api.start_flower_client",
    "start_flower_server": "exareme2.worker.flower.starter.starter_api.start_flower_server",
    "stop_flower_server": "exareme2.worker.flower.cleanup.cleanup_api.stop_flower_server",
    "stop_flower_client": "exareme2.worker.flower.cleanup.cleanup_api.stop_flower_client",
    "garbage_collect": "exareme2.worker.flower.cleanup.cleanup_api.garbage_collect",
    "run_exaflow_udf": "exareme2.worker.exaflow.udf.udf_api.run_udf",
    "run_yesql": "exareme2.worker.exaflow.yesql.yesql_api.run_yesql",
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

    def _get_celery_app(self):
        return CeleryAppFactory().get_celery_app(socket_addr=self._worker_queue_addr)

    def _queue_task(self, task_signature: str, **task_params) -> WorkerTaskResult:
        """
        Queues a task with the given signature and parameters.

        :param task_signature: The signature of the task to queue.
        :param task_params: A dictionary of parameters to pass to the task.
        :return: A WorkerTaskResult instance.
        """
        async_result = self._get_celery_app().queue_task(
            task_signature=task_signature,
            logger=self._logger,
            **task_params,
        )
        return WorkerTaskResult(self._get_celery_app(), async_result, self._logger)

    def get_tables(self, request_id: str, context_id: str) -> WorkerTaskResult:
        return self._queue_task(
            task_signature=TASK_SIGNATURES["get_tables"],
            request_id=request_id,
            context_id=context_id,
        )

    def get_table_data(self, request_id: str, table_name: str) -> WorkerTaskResult:
        return self._queue_task(
            task_signature=TASK_SIGNATURES["get_table_data"],
            request_id=request_id,
            table_name=table_name,
        )

    def create_table(
        self, request_id: str, context_id: str, command_id: str, schema: TableSchema
    ) -> WorkerTaskResult:
        return self._queue_task(
            task_signature=TASK_SIGNATURES["create_table"],
            request_id=request_id,
            context_id=context_id,
            command_id=command_id,
            schema_json=schema.json(),
        )

    def get_views(self, request_id: str, context_id: str) -> WorkerTaskResult:
        return self._queue_task(
            task_signature=TASK_SIGNATURES["get_views"],
            request_id=request_id,
            context_id=context_id,
        )

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
        return self._queue_task(
            task_signature=TASK_SIGNATURES["create_data_model_views"],
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

    def get_merge_tables(self, request_id: str, context_id: str) -> WorkerTaskResult:
        return self._queue_task(
            task_signature=TASK_SIGNATURES["get_merge_tables"],
            request_id=request_id,
            context_id=context_id,
        )

    def create_merge_table(
        self,
        request_id: str,
        context_id: str,
        command_id: str,
        table_infos: List[TableInfo],
    ) -> WorkerTaskResult:
        return self._queue_task(
            task_signature=TASK_SIGNATURES["create_merge_table"],
            command_id=command_id,
            request_id=request_id,
            context_id=context_id,
            table_infos_json=[table_info.json() for table_info in table_infos],
        )

    def get_remote_tables(self, request_id: str, context_id: str) -> WorkerTaskResult:
        return self._queue_task(
            task_signature=TASK_SIGNATURES["get_remote_tables"],
            request_id=request_id,
            context_id=context_id,
        )

    def create_remote_table(
        self,
        request_id: str,
        table_name: str,
        table_schema: TableSchema,
        monetdb_socket_address: str,
    ) -> WorkerTaskResult:
        return self._queue_task(
            task_signature=TASK_SIGNATURES["create_remote_table"],
            table_name=table_name,
            table_schema_json=table_schema.json(),
            monetdb_socket_address=monetdb_socket_address,
            request_id=request_id,
        )

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
        return self._queue_task(
            task_signature=TASK_SIGNATURES["run_udf"],
            command_id=command_id,
            request_id=request_id,
            context_id=context_id,
            func_name=func_name,
            positional_args_json=positional_args.json(),
            keyword_args_json=keyword_args.json(),
            use_smpc=use_smpc,
            output_schema=output_schema.json() if output_schema else None,
        )

    def validate_smpc_templates_match(
        self,
        request_id: str,
        table_name: str,
    ):
        return self._queue_task(
            task_signature=TASK_SIGNATURES["validate_smpc_templates_match"],
            request_id=request_id,
            table_name=table_name,
        )

    def load_data_to_smpc_client(
        self, request_id: str, table_name: str, jobid: str
    ) -> WorkerTaskResult:
        return self._queue_task(
            task_signature=TASK_SIGNATURES["load_data_to_smpc_client"],
            request_id=request_id,
            table_name=table_name,
            jobid=jobid,
        )

    def get_smpc_result(
        self,
        request_id: str,
        jobid: str,
        context_id: str,
        command_id: str,
        command_subid: Optional[str] = "0",
    ) -> WorkerTaskResult:
        return self._queue_task(
            task_signature=TASK_SIGNATURES["get_smpc_result"],
            request_id=request_id,
            jobid=jobid,
            context_id=context_id,
            command_id=command_id,
            command_subid=command_subid,
        )

    # CLEANUP functionality
    def queue_cleanup(self, request_id: str, context_id: str):
        return self._queue_task(
            task_signature=TASK_SIGNATURES["cleanup"],
            request_id=request_id,
            context_id=context_id,
        )

    def queue_worker_info_task(self, request_id: str) -> WorkerTaskResult:
        return self._queue_task(
            task_signature=TASK_SIGNATURES["get_worker_info"],
            request_id=request_id,
            priority=CELERY_APP_QUEUE_MAX_PRIORITY,
        )

    def queue_worker_datasets_per_data_model_task(
        self, request_id: str
    ) -> WorkerTaskResult:
        return self._queue_task(
            task_signature=TASK_SIGNATURES["get_worker_datasets_per_data_model"],
            request_id=request_id,
            priority=CELERY_APP_QUEUE_MAX_PRIORITY,
        )

    def queue_data_model_cdes_task(
        self, request_id: str, data_model: str
    ) -> WorkerTaskResult:
        return self._queue_task(
            task_signature=TASK_SIGNATURES["get_data_model_cdes"],
            request_id=request_id,
            data_model=data_model,
            priority=CELERY_APP_QUEUE_MAX_PRIORITY,
        )

    def queue_data_model_attributes_task(
        self, request_id: str, data_model: str
    ) -> WorkerTaskResult:
        return self._queue_task(
            task_signature=TASK_SIGNATURES["get_data_model_attributes"],
            request_id=request_id,
            data_model=data_model,
            priority=CELERY_APP_QUEUE_MAX_PRIORITY,
        )

    # --------------- healthcheck task ---------------
    # NON-BLOCKING
    def queue_healthcheck_task(
        self, request_id: str, check_db: bool
    ) -> WorkerTaskResult:
        return self._queue_task(
            task_signature=TASK_SIGNATURES["healthcheck"],
            request_id=request_id,
            check_db=check_db,
            priority=CELERY_APP_QUEUE_MAX_PRIORITY,
        )

    def start_flower_client(
        self,
        request_id,
        algorithm_folder_path,
        server_address,
        data_model,
        datasets,
        execution_timeout,
    ) -> WorkerTaskResult:
        return self._queue_task(
            task_signature=TASK_SIGNATURES["start_flower_client"],
            request_id=request_id,
            algorithm_folder_path=algorithm_folder_path,
            server_address=server_address,
            data_model=data_model,
            datasets=datasets,
            execution_timeout=execution_timeout,
        )

    def start_flower_server(
        self,
        request_id,
        algorithm_folder_path,
        number_of_clients,
        server_address,
        data_model,
        datasets,
    ) -> WorkerTaskResult:
        return self._queue_task(
            task_signature=TASK_SIGNATURES["start_flower_server"],
            request_id=request_id,
            algorithm_folder_path=algorithm_folder_path,
            number_of_clients=number_of_clients,
            server_address=server_address,
            data_model=data_model,
            datasets=datasets,
        )

    def stop_flower_server(
        self, request_id, pid: int, algorithm_name: str
    ) -> WorkerTaskResult:
        return self._queue_task(
            task_signature=TASK_SIGNATURES["stop_flower_server"],
            request_id=request_id,
            pid=pid,
            algorithm_name=algorithm_name,
        )

    def stop_flower_client(
        self, request_id, pid: int, algorithm_name: str
    ) -> WorkerTaskResult:
        return self._queue_task(
            task_signature=TASK_SIGNATURES["stop_flower_client"],
            request_id=request_id,
            pid=pid,
            algorithm_name=algorithm_name,
        )

    def garbage_collect(self, request_id) -> WorkerTaskResult:
        return self._queue_task(
            task_signature=TASK_SIGNATURES["garbage_collect"],
            request_id=request_id,
        )

    def queue_udf(
        self,
        request_id,
        udf_registry_key,
        params,
    ) -> WorkerTaskResult:
        return self._queue_task(
            task_signature=TASK_SIGNATURES["run_exaflow_udf"],
            request_id=request_id,
            udf_registry_key=udf_registry_key,
            params=params,
        )

    def queue_yesql(
        self,
        request_id,
        udf_registry_key,
        params,
    ) -> WorkerTaskResult:
        return self._queue_task(
            task_signature=TASK_SIGNATURES["run_yesql"],
            request_id=request_id,
            udf_registry_key=udf_registry_key,
            params=params,
        )
