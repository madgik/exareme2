from typing import Callable
from typing import Final
from typing import List
from typing import Optional
from typing import Tuple

from billiard.exceptions import SoftTimeLimitExceeded
from billiard.exceptions import TimeLimitExceeded
from celery.exceptions import TimeoutError
from celery.result import AsyncResult
from kombu.exceptions import OperationalError

from mipengine.controller.celery_app import get_node_celery_app
from mipengine.controller.node_tasks_handler_interface import INodeTasksHandler
from mipengine.controller.node_tasks_handler_interface import IQueuedUDFAsyncResult
from mipengine.controller.node_tasks_handler_interface import UDFKeyArguments
from mipengine.controller.node_tasks_handler_interface import UDFPosArguments
from mipengine.node_tasks_DTOs import TableData
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import UDFResults

TASK_SIGNATURES: Final = {
    "get_tables": "mipengine.node.tasks.tables.get_tables",
    "get_table_schema": "mipengine.node.tasks.common.get_table_schema",
    "get_table_data": "mipengine.node.tasks.common.get_table_data",
    "create_table": "mipengine.node.tasks.tables.create_table",
    "get_views": "mipengine.node.tasks.views.get_views",
    "create_data_model_view": "mipengine.node.tasks.views.create_data_model_view",
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


def time_limit_exceeded_handler(method: Callable):
    def inner(ref, *args, **kwargs):
        # the following exceptions are thrown when the get() method is called
        # on an AsyncResult.
        # SoftTimeLimitExceeded and TimeLimitExceeded are raised by the celery app
        # instantiating the workers that execute the task (aka the node's celery app)
        # and propagated here, whereas the
        # TimeoutError exception is raised by the caller celery app (aka the
        # NodeTasksHandlerCelery), when the get() method has not returned after a time
        # limit is exceeded. The time limit is independently defined on the caller
        # celery app (the NodeTasksHandlerCelery) and executing celery app (the node)
        try:
            return method(ref, *args, **kwargs)
        except SoftTimeLimitExceeded:
            # TODO should use kwargs here..
            raise SoftTimeLimitExceeded(
                {"node_id": ref.node_id, "task": method.__name__, "args": args}
            )
        except TimeLimitExceeded:
            # TODO should use kwargs here..
            raise TimeLimitExceeded(
                {"node_id": ref.node_id, "task": method.__name__, "args": args}
            )
        except TimeoutError:
            # TODO should use kwargs here..
            raise TimeoutError(
                {
                    "node_id": ref.node_id,
                    "task": method.__name__,
                    "args": args,
                    "kwargs": kwargs,
                }
            )

    return inner


def broker_connection_closed_handler(method: Callable):
    def inner(ref, *args, **kwargs):
        try:
            return method(ref, *args, **kwargs)
        except (OperationalError, ConnectionResetError):
            raise ClosedBrokerConnectionError(
                message=f"Connection to broker closed for node:{ref.node_id} when tried "
                f"to call {method} with task_kwargs={kwargs}",
                node_id=ref.node_id,
            )

    return inner


class QueuedUDFAsyncResult(IQueuedUDFAsyncResult):
    @time_limit_exceeded_handler
    @broker_connection_closed_handler
    def get(self, timeout=None):
        return self.async_result.get(timeout)


class ClosedBrokerConnectionError(Exception):
    def __init__(self, node_id: str, message: str):
        self.node_id = node_id
        super().__init__(message)


class NodeTasksHandlerCelery(INodeTasksHandler):

    # TODO create custom type and validator for the socket address
    def __init__(
        self, node_id: str, node_queue_addr: str, node_db_addr: str, tasks_timeout
    ):
        self._node_id = node_id
        self._celery_app = get_node_celery_app(node_queue_addr)
        self._db_address = node_db_addr
        self._tasks_timeout = tasks_timeout

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def node_data_address(self) -> str:
        return self._db_address

    def _apply_async(self, task_signature, **kwargs) -> AsyncResult:
        # The existing connection to the broker is passed in apply_async because the
        # default behaviour (not passing a
        # connection object), when the broker is down, is for the celery app to try to
        # create a new connection to the broker, without raising any exceptions.
        # Nevertheless while the broker is down the call to apply_async just hangs
        # waiting for a connection with the broker to be established. Passing the
        # existing connection object to apply_async causes the call to raise an
        # exception if the broker is down
        async_result = task_signature.apply_async(
            connection=self._celery_app.broker_connection(), kwargs=kwargs
        )

        return async_result

    # TABLES functionality
    @time_limit_exceeded_handler
    @broker_connection_closed_handler
    def get_tables(self, request_id: str, context_id: str) -> List[str]:
        task_signature = self._celery_app.signature(TASK_SIGNATURES["get_tables"])
        result = self._apply_async(
            task_signature=task_signature, request_id=request_id, context_id=context_id
        ).get(self._tasks_timeout)
        return list(result)

    @time_limit_exceeded_handler
    @broker_connection_closed_handler
    def get_table_schema(self, request_id, table_name: str) -> TableSchema:
        task_signature = self._celery_app.signature(TASK_SIGNATURES["get_table_schema"])
        result = self._apply_async(
            task_signature=task_signature, request_id=request_id, table_name=table_name
        ).get(self._tasks_timeout)
        return TableSchema.parse_raw(result)

    @time_limit_exceeded_handler
    @broker_connection_closed_handler
    def get_table_data(self, request_id, table_name: str) -> TableData:
        task_signature = self._celery_app.signature(TASK_SIGNATURES["get_table_data"])
        result = self._apply_async(
            task_signature=task_signature, request_id=request_id, table_name=table_name
        ).get(self._tasks_timeout)
        return TableData.parse_raw(result)

    @time_limit_exceeded_handler
    @broker_connection_closed_handler
    def create_table(
        self, request_id: str, context_id: str, command_id: str, schema: TableSchema
    ) -> str:
        schema_json = schema.json()
        task_signature = self._celery_app.signature(TASK_SIGNATURES["create_table"])
        result = self._apply_async(
            task_signature=task_signature,
            request_id=request_id,
            context_id=context_id,
            command_id=command_id,
            schema_json=schema_json,
        ).get(self._tasks_timeout)
        return result

    # VIEWS functionality
    @time_limit_exceeded_handler
    @broker_connection_closed_handler
    def get_views(self, request_id: str, context_id: str) -> List[str]:
        task_signature = self._celery_app.signature(TASK_SIGNATURES["get_views"])
        result = self._apply_async(
            task_signature=task_signature, request_id=request_id, context_id=context_id
        ).get(self._tasks_timeout)
        return result

    # TODO: this is very specific to mip, very inconsistent with the rest, has to be abstracted somehow
    @time_limit_exceeded_handler
    @broker_connection_closed_handler
    def create_data_model_view(
        self,
        request_id: str,
        context_id: str,
        command_id: str,
        data_model: str,
        columns: List[str],
        filters: List[str],
    ) -> str:
        task_signature = self._celery_app.signature(
            TASK_SIGNATURES["create_data_model_view"]
        )
        result = self._apply_async(
            task_signature=task_signature,
            request_id=request_id,
            context_id=context_id,
            command_id=command_id,
            data_model=data_model,
            columns=columns,
            filters=filters,
        ).get(self._tasks_timeout)
        return result

    # MERGE TABLES functionality
    @time_limit_exceeded_handler
    @broker_connection_closed_handler
    def get_merge_tables(self, request_id: str, context_id: str) -> List[str]:
        task_signature = self._celery_app.signature(TASK_SIGNATURES["get_merge_tables"])
        result = self._apply_async(
            task_signature=task_signature, request_id=request_id, context_id=context_id
        ).get(self._tasks_timeout)
        return result

    @time_limit_exceeded_handler
    @broker_connection_closed_handler
    def create_merge_table(
        self, request_id: str, context_id: str, command_id: str, table_names: List[str]
    ):
        task_signature = self._celery_app.signature(
            TASK_SIGNATURES["create_merge_table"]
        )
        result = self._apply_async(
            task_signature=task_signature,
            command_id=command_id,
            request_id=request_id,
            context_id=context_id,
            table_names=table_names,
        ).get(self._tasks_timeout)
        return result

    # REMOTE TABLES functionality
    @time_limit_exceeded_handler
    @broker_connection_closed_handler
    def get_remote_tables(self, request_id: str, context_id: str) -> List[str]:
        task_signature = self._celery_app.signature(
            TASK_SIGNATURES["get_remote_tables"]
        )
        return self._apply_async(
            task_signature=task_signature, request_id=request_id, context_id=context_id
        ).get(self._tasks_timeout)

    @time_limit_exceeded_handler
    @broker_connection_closed_handler
    def create_remote_table(
        self,
        request_id,
        table_name: str,
        table_schema: TableSchema,
        original_db_url: str,
    ):
        table_schema_json = table_schema.json()
        task_signature = self._celery_app.signature(
            TASK_SIGNATURES["create_remote_table"]
        )
        self._apply_async(
            task_signature=task_signature,
            table_name=table_name,
            table_schema_json=table_schema_json,
            monetdb_socket_address=original_db_url,
            request_id=request_id,
        ).get(self._tasks_timeout)

    # UDFs functionality
    @broker_connection_closed_handler
    def queue_run_udf(
        self,
        request_id: str,
        context_id: str,
        command_id: str,
        func_name: str,
        positional_args: UDFPosArguments,
        keyword_args: UDFKeyArguments,
        use_smpc: bool = False,
    ) -> QueuedUDFAsyncResult:

        task_signature = self._celery_app.signature(TASK_SIGNATURES["run_udf"])
        async_result = self._apply_async(
            task_signature=task_signature,
            command_id=command_id,
            request_id=request_id,
            context_id=context_id,
            func_name=func_name,
            positional_args_json=positional_args.json(),
            keyword_args_json=keyword_args.json(),
            use_smpc=use_smpc,
        )
        return QueuedUDFAsyncResult(
            node_id=self.node_id,
            command_id=command_id,
            request_id=request_id,
            context_id=context_id,
            func_name=func_name,
            positional_args=positional_args,
            keyword_args=keyword_args,
            use_smpc=use_smpc,
            async_result=async_result,
        )

    @time_limit_exceeded_handler
    @broker_connection_closed_handler
    def get_queued_udf_result(self, async_result: QueuedUDFAsyncResult) -> UDFResults:
        result_str = async_result.get(self._tasks_timeout)
        return UDFResults.parse_raw(result_str)

    @time_limit_exceeded_handler
    @broker_connection_closed_handler
    def get_udfs(self, algorithm_name) -> List[str]:
        task_signature = self._celery_app.signature(TASK_SIGNATURES["get_udfs"])
        result = self._apply_async(
            task_signature=task_signature, algorithm_name=algorithm_name
        ).get(self._tasks_timeout)
        return result

    # return the generated monetdb pythonudf
    @time_limit_exceeded_handler
    @broker_connection_closed_handler
    def get_run_udf_query(
        self,
        request_id: str,
        context_id: str,
        command_id: str,
        func_name: str,
        positional_args: UDFPosArguments,
    ) -> Tuple[str, str]:
        task_signature = self._celery_app.signature(
            TASK_SIGNATURES["get_run_udf_query"]
        )
        result = self._apply_async(
            task_signature=task_signature,
            command_id=command_id,
            request_id=request_id,
            context_id=context_id,
            func_name=func_name,
            positional_args_json=positional_args.json(),
            keyword_args_json=UDFKeyArguments(args={}).json(),
        ).get(self._tasks_timeout)
        return result

    # ------------- SMPC functionality ---------------
    @time_limit_exceeded_handler
    @broker_connection_closed_handler
    def validate_smpc_templates_match(
        self,
        context_id: str,
        table_name: str,
    ):
        task_signature = self._celery_app.signature(
            TASK_SIGNATURES["validate_smpc_templates_match"]
        )
        self._apply_async(
            task_signature=task_signature,
            context_id=context_id,
            table_name=table_name,
        ).get(self._tasks_timeout)

    @time_limit_exceeded_handler
    @broker_connection_closed_handler
    def load_data_to_smpc_client(
        self, context_id: str, table_name: str, jobid: str
    ) -> int:
        task_signature = self._celery_app.signature(
            TASK_SIGNATURES["load_data_to_smpc_client"]
        )
        return self._apply_async(
            task_signature=task_signature,
            context_id=context_id,
            table_name=table_name,
            jobid=jobid,
        ).get(self._tasks_timeout)

    @time_limit_exceeded_handler
    @broker_connection_closed_handler
    def get_smpc_result(
        self,
        jobid: str,
        context_id: str,
        command_id: str,
        command_subid: Optional[str] = "0",
    ) -> str:
        task_signature = self._celery_app.signature(TASK_SIGNATURES["get_smpc_result"])
        return self._apply_async(
            task_signature=task_signature,
            jobid=jobid,
            context_id=context_id,
            command_id=command_id,
            command_subid=command_subid,
        ).get(self._tasks_timeout)

    # CLEANUP functionality
    @time_limit_exceeded_handler
    @broker_connection_closed_handler
    def clean_up(self, request_id: str, context_id: str):
        task_signature = self._celery_app.signature(TASK_SIGNATURES["clean_up"])
        self._apply_async(
            task_signature=task_signature, request_id=request_id, context_id=context_id
        ).get(self._tasks_timeout)
