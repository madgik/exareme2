from abc import ABC
from abc import abstractmethod
from typing import Final
from typing import List
from typing import Optional
from typing import Tuple

from celery.result import AsyncResult

from mipengine.controller.celery_app import CeleryAppFactory
from mipengine.node_tasks_DTOs import TableData
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import UDFKeyArguments
from mipengine.node_tasks_DTOs import UDFPosArguments
from mipengine.node_tasks_DTOs import UDFResults

TASK_SIGNATURES: Final = {
    "get_tables": "mipengine.node.tasks.tables.get_tables",
    "get_table_schema": "mipengine.node.tasks.common.get_table_schema",
    "get_table_data": "mipengine.node.tasks.common.get_table_data",
    "create_table": "mipengine.node.tasks.tables.create_table",
    "get_views": "mipengine.node.tasks.views.get_views",
    "create_data_model_views": "mipengine.node.tasks.views.create_data_model_views",
    "get_remote_tables": "mipengine.node.tasks.remote_tables.get_remote_tables",
    "create_remote_table": "mipengine.node.tasks.remote_tables.create_remote_table",
    "get_merge_tables": "mipengine.node.tasks.merge_tables.get_merge_tables",
    "create_merge_table": "mipengine.node.tasks.merge_tables.create_merge_table",
    "get_udfs": "mipengine.node.tasks.udfs.get_udfs",
    "run_udf": "mipengine.node.tasks.udfs.run_udf",
    "get_run_udf_query": "mipengine.node.tasks.udfs.get_run_udf_query",
    "clean_up": "mipengine.node.tasks.common.clean_up",
    "validate_smpc_templates_match": "mipengine.node.tasks.smpc.validate_smpc_templates_match",
    "load_data_to_smpc_client": "mipengine.node.tasks.smpc.load_data_to_smpc_client",
    "get_smpc_result": "mipengine.node.tasks.smpc.get_smpc_result",
}


class INodeAlgorithmTasksHandler(ABC):
    @property
    @abstractmethod
    def node_id(self) -> str:
        pass

    @property
    @abstractmethod
    def node_data_address(self) -> str:
        pass

    @property
    @abstractmethod
    def tasks_timeout(self) -> int:
        pass

    # TABLES functionality
    @abstractmethod
    def get_tables(self, request_id: str, context_id: str) -> List[str]:
        pass

    @abstractmethod
    def get_table_schema(self, request_id: str, table_name: str):
        pass

    @abstractmethod
    def get_table_data(self, request_id: str, table_name: str) -> TableData:
        pass

    @abstractmethod
    def create_table(
        self, request_id: str, context_id: str, command_id: str, schema: TableSchema
    ) -> str:
        pass

    # VIEWS functionality
    @abstractmethod
    def get_views(self, request_id: str, context_id: str) -> List[str]:
        pass

    @abstractmethod
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
    ) -> List[str]:
        pass

    # MERGE TABLES functionality
    @abstractmethod
    def get_merge_tables(self, request_id: str, context_id: str) -> List[str]:
        pass

    @abstractmethod
    def create_merge_table(
        self, request_id: str, context_id: str, command_id: str, table_names: List[str]
    ):
        pass

    # REMOTE TABLES functionality
    @abstractmethod
    def get_remote_tables(self, request_id: str, context_id: str) -> List[str]:
        pass

    @abstractmethod
    def create_remote_table(
        self,
        request_id: str,
        table_name: str,
        table_schema: TableSchema,
        original_db_url: str,
    ) -> str:
        pass

    # UDFs functionality
    @abstractmethod
    def queue_run_udf(
        self,
        request_id: str,
        context_id: str,
        command_id: str,
        func_name: str,
        positional_args: Optional[UDFPosArguments] = None,
        keyword_args: Optional[UDFKeyArguments] = None,
        use_smpc: bool = False,
    ) -> AsyncResult:
        pass

    @abstractmethod
    def get_queued_udf_result(
        self, async_result: AsyncResult, request_id: str
    ) -> UDFResults:
        pass

    @abstractmethod
    def get_udfs(self, request_id: str, algorithm_name) -> List[str]:
        pass

    # return the generated monetdb python udf
    @abstractmethod
    def get_run_udf_query(
        self,
        request_id: str,
        context_id: str,
        command_id: str,
        func_name: str,
        positional_args: UDFPosArguments,
    ) -> Tuple[str, str]:
        pass

    # CLEANUP functionality
    @abstractmethod
    def clean_up(self, request_id: str, context_id: str):
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
    ) -> int:
        pass

    @abstractmethod
    def get_smpc_result(
        self,
        jobid: str,
        context_id: str,
        command_id: str,
        command_subid: Optional[str] = "0",
    ) -> str:
        pass


class NodeAlgorithmTasksHandler(INodeAlgorithmTasksHandler):

    # TODO create custom type and validator for the socket address
    def __init__(
        self, node_id: str, node_queue_addr: str, node_db_addr: str, tasks_timeout
    ):
        self._node_id = node_id
        self._node_queue_addr = node_queue_addr
        self._db_address = node_db_addr
        self._tasks_timeout = tasks_timeout

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def node_data_address(self) -> str:
        return self._db_address

    @property
    def tasks_timeout(self) -> int:
        return self._tasks_timeout

    def _get_node_celery_app(self):
        return CeleryAppFactory().get_celery_app(socket_addr=self._node_queue_addr)

    # TABLES functionality
    def get_tables(self, request_id: str, context_id: str) -> List[str]:
        celery_app = self._get_node_celery_app()
        task_signature = TASK_SIGNATURES["get_tables"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            request_id=request_id,
            context_id=context_id,
        )
        result = celery_app.get_result(
            async_result=async_result,
            timeout=self._tasks_timeout,
            request_id=request_id,
        )
        return list(result)

    def get_table_schema(self, request_id, table_name: str) -> TableSchema:
        celery_app = self._get_node_celery_app()
        task_signature = TASK_SIGNATURES["get_table_schema"]
        async_result = celery_app.queue_task(
            task_signature=task_signature, request_id=request_id, table_name=table_name
        )
        result = celery_app.get_result(
            async_result=async_result,
            timeout=self._tasks_timeout,
            request_id=request_id,
        )
        return TableSchema.parse_raw(result)

    def get_table_data(self, request_id, table_name: str) -> TableData:
        celery_app = self._get_node_celery_app()
        task_signature = TASK_SIGNATURES["get_table_data"]
        async_result = celery_app.queue_task(
            task_signature=task_signature, request_id=request_id, table_name=table_name
        )
        result = celery_app.get_result(
            async_result=async_result,
            timeout=self._tasks_timeout,
            request_id=request_id,
        )
        return TableData.parse_raw(result)

    def create_table(
        self, request_id: str, context_id: str, command_id: str, schema: TableSchema
    ) -> str:
        schema_json = schema.json()

        celery_app = self._get_node_celery_app()
        task_signature = TASK_SIGNATURES["create_table"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            request_id=request_id,
            context_id=context_id,
            command_id=command_id,
            schema_json=schema_json,
        )
        result = celery_app.get_result(
            async_result=async_result,
            timeout=self._tasks_timeout,
            request_id=request_id,
        )

        return result

    # VIEWS functionality
    def get_views(self, request_id: str, context_id: str) -> List[str]:
        celery_app = self._get_node_celery_app()
        task_signature = TASK_SIGNATURES["get_views"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            request_id=request_id,
            context_id=context_id,
        )
        result = celery_app.get_result(
            async_result=async_result,
            timeout=self._tasks_timeout,
            request_id=request_id,
        )

        return result

    # TODO: this is very specific to mip, very inconsistent with the rest, has to be abstracted somehow
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
    ) -> str:
        celery_app = self._get_node_celery_app()
        task_signature = TASK_SIGNATURES["create_data_model_views"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
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
        result = celery_app.get_result(
            async_result=async_result,
            timeout=self._tasks_timeout,
            request_id=request_id,
        )
        return result

    # MERGE TABLES functionality
    def get_merge_tables(self, request_id: str, context_id: str) -> List[str]:
        celery_app = self._get_node_celery_app()
        task_signature = TASK_SIGNATURES["get_merge_tables"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            request_id=request_id,
            context_id=context_id,
        )
        result = celery_app.get_result(
            async_result=async_result,
            timeout=self._tasks_timeout,
            request_id=request_id,
        )

        return result

    def create_merge_table(
        self, request_id: str, context_id: str, command_id: str, table_names: List[str]
    ):
        celery_app = self._get_node_celery_app()
        task_signature = TASK_SIGNATURES["create_merge_table"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            command_id=command_id,
            request_id=request_id,
            context_id=context_id,
            table_names=table_names,
        )
        result = celery_app.get_result(
            async_result=async_result,
            timeout=self._tasks_timeout,
            request_id=request_id,
        )

        return result

    # REMOTE TABLES functionality
    def get_remote_tables(self, request_id: str, context_id: str) -> List[str]:
        celery_app = self._get_node_celery_app()
        task_signature = TASK_SIGNATURES["get_remote_tables"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            request_id=request_id,
            context_id=context_id,
        )
        result = celery_app.get_result(
            async_result=async_result,
            timeout=self._tasks_timeout,
            request_id=request_id,
        )
        return result

    def create_remote_table(
        self,
        request_id,
        table_name: str,
        table_schema: TableSchema,
        original_db_url: str,
    ):
        table_schema_json = table_schema.json()

        celery_app = self._get_node_celery_app()
        task_signature = TASK_SIGNATURES["create_remote_table"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            table_name=table_name,
            table_schema_json=table_schema_json,
            monetdb_socket_address=original_db_url,
            request_id=request_id,
        )
        celery_app.get_result(
            async_result=async_result,
            timeout=self._tasks_timeout,
            request_id=request_id,
        )

    # UDFs functionality
    def queue_run_udf(
        self,
        request_id: str,
        context_id: str,
        command_id: str,
        func_name: str,
        positional_args: UDFPosArguments,
        keyword_args: UDFKeyArguments,
        use_smpc: bool = False,
    ) -> AsyncResult:

        celery_app = self._get_node_celery_app()
        task_signature = TASK_SIGNATURES["run_udf"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            command_id=command_id,
            request_id=request_id,
            context_id=context_id,
            func_name=func_name,
            positional_args_json=positional_args.json(),
            keyword_args_json=keyword_args.json(),
            use_smpc=use_smpc,
        )
        return async_result

    def get_queued_udf_result(
        self,
        async_result: AsyncResult,
        request_id: str,
    ) -> UDFResults:
        celery_app = self._get_node_celery_app()
        result = celery_app.get_result(
            async_result=async_result,
            timeout=self._tasks_timeout,
            request_id=request_id,
        )
        return UDFResults.parse_raw(result)

    def get_udfs(self, algorithm_name) -> List[str]:
        celery_app = self._get_node_celery_app()
        task_signature = TASK_SIGNATURES["get_udfs"]
        async_result = celery_app.queue_task(
            task_signature=task_signature, algorithm_name=algorithm_name
        )
        result = celery_app.get_result(
            async_result=async_result,
            timeout=self._tasks_timeout,
            request_id=request_id,
        )
        return result

    # return the generated monetdb pythonudf
    def get_run_udf_query(
        self,
        request_id: str,
        context_id: str,
        command_id: str,
        func_name: str,
        positional_args: UDFPosArguments,
    ) -> Tuple[str, str]:
        celery_app = self._get_node_celery_app()
        task_signature = TASK_SIGNATURES["get_run_udf_query"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            command_id=command_id,
            request_id=request_id,
            context_id=context_id,
            func_name=func_name,
            positional_args_json=positional_args.json(),
            keyword_args_json=UDFKeyArguments(args={}).json(),
        )
        result = celery_app.get_result(
            async_result=async_result,
            timeout=self._tasks_timeout,
            request_id=request_id,
        )

        return result

    # ------------- SMPC functionality ---------------
    def validate_smpc_templates_match(
        self,
        request_id: str,
        table_name: str,
    ):
        celery_app = self._get_node_celery_app()
        task_signature = TASK_SIGNATURES["validate_smpc_templates_match"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            request_id=request_id,
            table_name=table_name,
        )
        celery_app.get_result(
            async_result=async_result,
            timeout=self._tasks_timeout,
            request_id=request_id,
        )

    def load_data_to_smpc_client(
        self, request_id: str, table_name: str, jobid: str
    ) -> int:
        celery_app = self._get_node_celery_app()
        task_signature = TASK_SIGNATURES["load_data_to_smpc_client"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            request_id=request_id,
            table_name=table_name,
            jobid=jobid,
        )
        result = celery_app.get_result(
            async_result=async_result,
            timeout=self._tasks_timeout,
            request_id=request_id,
        )
        return result

    def get_smpc_result(
        self,
        request_id: str,
        jobid: str,
        context_id: str,
        command_id: str,
        command_subid: Optional[str] = "0",
    ) -> str:
        celery_app = self._get_node_celery_app()
        task_signature = TASK_SIGNATURES["get_smpc_result"]
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            request_id=request_id,
            jobid=jobid,
            context_id=context_id,
            command_id=command_id,
            command_subid=command_subid,
        )
        celery_app.get_result(
            async_result=async_result,
            timeout=self._tasks_timeout,
            request_id=request_id,
        )

    # CLEANUP functionality
    def clean_up(self, request_id: str, context_id: str):
        celery_app = self._get_node_celery_app()
        task_signature = TASK_SIGNATURES["clean_up"]
        async_result = celery_app.queue_task(
            task_signature=task_signature, request_id=request_id, context_id=context_id
        )
        celery_app.get_result(
            async_result=async_result,
            timeout=self._tasks_timeout,
            request_id=request_id,
        )
