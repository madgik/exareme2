from mipengine.controller.node_tasks_handler_interface import INodeTasksHandler
from mipengine.controller.node_tasks_handler_interface import IQueuedUDFAsyncResult
from pydantic import BaseModel, conint
from ipaddress import IPv4Address
from celery import Celery

from celery.result import AsyncResult

from typing import List, Tuple, Final, Callable, Dict, Any

from celery.exceptions import TimeoutError
from billiard.exceptions import SoftTimeLimitExceeded
from billiard.exceptions import TimeLimitExceeded
from kombu.exceptions import OperationalError

from mipengine.controller import config as controller_config
from mipengine.node_tasks_DTOs import TableData
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import TableInfo

TASK_SIGNATURES: Final = {
    "get_tables": "mipengine.node.tasks.tables.get_tables",
    "get_table_schema": "mipengine.node.tasks.common.get_table_schema",
    "get_table_data": "mipengine.node.tasks.common.get_table_data",
    "create_table": "mipengine.node.tasks.tables.create_table",
    "get_views": "mipengine.node.tasks.views.get_views",
    "create_pathology_view": "mipengine.node.tasks.views.create_pathology_view",
    "get_remote_tables": "mipengine.node.tasks.remote_tables.get_remote_tables",
    "create_remote_table": "mipengine.node.tasks.remote_tables.create_remote_table",
    "get_merge_tables": "mipengine.node.tasks.merge_tables.get_merge_tables",
    "create_merge_table": "mipengine.node.tasks.merge_tables.create_merge_table",
    "get_udfs": "mipengine.node.tasks.udfs.get_udfs",
    "run_udf": "mipengine.node.tasks.udfs.run_udf",
    "get_run_udf_query": "mipengine.node.tasks.udfs.get_run_udf_query",
    "clean_up": "mipengine.node.tasks.common.clean_up",
}


class QueuedUDFAsyncResult(IQueuedUDFAsyncResult):
    def get(self, timeout=None):
        try:
            return self.async_result.get(timeout)
        except (OperationalError, ConnectionResetError) as err:
            raise ClosedBrokerConnectionError(
                message=f"Connection to broker closed for {self.node_id=} "
                f"while tried to get result of {self}",
                node_id=self.node_id,
            )


class CeleryParamsDTO(BaseModel):
    task_queue_domain: IPv4Address
    task_queue_port: conint(ge=1024, le=65535)

    db_domain: IPv4Address
    db_port: conint(ge=1024, le=65535)

    user: str
    password: str
    vhost: str

    tasks_timeout: conint(ge=0)
    max_retries: conint(ge=0)
    interval_start: conint(ge=0)
    interval_step: conint(ge=0)
    interval_max: conint(ge=0)


def time_limit_exceeded_handler(method: Callable):
    def inner(ref, *args, **kwargs):
        # the following exceptions are thrown when the get() method is called
        # on an AsyncResult. SoftTimeLimitExceeded and TimeLimitExceeded are raised
        # by the calery app instantiating the workers executing the task (aka the
        # node's celery app) and propagated here, whereas the TimeoutError exception
        # is raised by the caller celery app (aka the NodeTasksHandlerCelery), when the
        # get() method has not returned after a time limit is exceeded. The time limit
        # is independently defined on the caller celery app (the NodeTasksHandlerCelery)
        # and executing celery app (the node)
        try:
            return method(ref, *args, **kwargs)
        except SoftTimeLimitExceeded as stle:
            # TODO should use kwargs here..
            raise SoftTimeLimitExceeded(
                {"node_id": ref.node_id, "task": method.__name__, "args": args}
            )
        except TimeLimitExceeded as tle:
            # TODO should use kwargs here..
            raise TimeLimitExceeded(
                {"node_id": ref.node_id, "task": method.__name__, "args": args}
            )
        except TimeoutError as te:
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
        except (OperationalError, ConnectionResetError) as err:
            raise ClosedBrokerConnectionError(
                message=f"Connection to broker closed for {ref.node_id=} while tried "
                f"to call task:{method} with task_kwargs={kwargs}",
                node_id=ref.node_id,
            )

    return inner


class ClosedBrokerConnectionError(Exception):
    def __init__(self, node_id: str, message: str):
        self.node_id = node_id
        super().__init__(message)


class NodeTasksHandlerCelery(INodeTasksHandler):

    # TODO create custom type and validator for the socket address
    def __init__(self, node_id: str, celery_params: "CeleryParamsDTO"):
        self._node_id = node_id

        user = celery_params.user
        password = celery_params.password

        queue_addr = ":".join(
            [str(celery_params.task_queue_domain), str(celery_params.task_queue_port)]
        )
        vhost = celery_params.vhost
        broker = f"amqp://{user}:{password}@{queue_addr}/{vhost}"

        celery_app = Celery(broker=broker, backend="rpc://")

        broker_transport_options = {
            "max_retries": celery_params.max_retries,
            "interval_start": celery_params.interval_start,
            "interval_step": celery_params.interval_step,
            "interval_max": celery_params.interval_max,
        }
        celery_app.conf.broker_transport_options = broker_transport_options

        self._celery_app = celery_app

        self._db_address = ":".join(
            [str(celery_params.db_domain), str(celery_params.db_port)]
        )

        self._task_timeout = celery_params.tasks_timeout

    @property
    def node_id(self):
        return self._node_id

    @property
    def node_data_address(self):
        return self._db_address

    def _apply_async(self, task_signature, **kwargs) -> AsyncResult:
        try:
            # the reason the existing connection to the broker is passed here as an
            # argument in apply_async is because the default behaviour (not passing a
            # connection object), when the broker is down, is for the celery app to try to
            # create a new connection to the broker, without raising any exceptions.
            # Nevertheless while the broker is down the call to apply_async just hangs
            # waiting for a connection with the broker to be established. Passing the
            # existing connection object to apply_async causes the call to raise an
            # exception if the broker is down
            async_result = task_signature.apply_async(
                connection=self._celery_app.broker_connection(), kwargs=kwargs
            )
        except (OperationalError, ConnectionResetError) as err:
            raise ClosedBrokerConnectionError(
                message=f"Connection to broker closed for {self.node_id=} while "
                f"tried to queue {task_signature}",
                node_id=self.node_id,
            )

        return async_result

    # TABLES functionality
    @time_limit_exceeded_handler
    @broker_connection_closed_handler
    def get_tables(self, context_id: str) -> List[str]:
        task_signature = self._celery_app.signature(TASK_SIGNATURES["get_tables"])
        result = self._apply_async(
            task_signature=task_signature, context_id=context_id
        ).get(self._task_timeout)
        return [table_name for table_name in result]

    @time_limit_exceeded_handler
    @broker_connection_closed_handler
    def get_table_schema(self, table_name: str):
        task_signature = self._celery_app.signature(TASK_SIGNATURES["get_table_schema"])
        result = self._apply_async(
            task_signature=task_signature, table_name=table_name
        ).get(self._task_timeout)
        return TableSchema.parse_raw(result)

    @time_limit_exceeded_handler
    @broker_connection_closed_handler
    def get_table_data(self, table_name: str) -> TableData:
        task_signature = self._celery_app.signature(TASK_SIGNATURES["get_table_data"])
        result = self._apply_async(
            task_signature=task_signature, table_name=table_name
        ).get(self._task_timeout)
        return TableData.parse_raw(result)

    @time_limit_exceeded_handler
    @broker_connection_closed_handler
    def create_table(
        self, context_id: str, command_id: str, schema: TableSchema
    ) -> str:
        schema_json = schema.json()
        task_signature = self._celery_app.signature(TASK_SIGNATURES["create_table"])
        result = self._apply_async(
            task_signature=task_signature,
            context_id=context_id,
            command_id=command_id,
            schema_json=schema_json,
        ).get(self._task_timeout)
        return result

    # VIEWS functionality
    @time_limit_exceeded_handler
    @broker_connection_closed_handler
    def get_views(self, context_id: str) -> List[str]:
        task_signature = self._celery_app.signature(TASK_SIGNATURES["get_views"])
        result = self._apply_async(
            task_signature=task_signature, context_id=context_id
        ).get(self._task_timeout)
        return result

    # TODO: this is very specific to mip, very inconsistent with the rest, has to be abstracted somehow
    @time_limit_exceeded_handler
    @broker_connection_closed_handler
    def create_pathology_view(
        self,
        context_id: str,
        command_id: str,
        pathology: str,
        columns: List[str],
        filters: List[str],
    ) -> str:
        task_signature = self._celery_app.signature(
            TASK_SIGNATURES["create_pathology_view"]
        )
        result = self._apply_async(
            task_signature=task_signature,
            context_id=context_id,
            command_id=command_id,
            pathology=pathology,
            columns=columns,
            filters=filters,
        ).get(self._task_timeout)

        return result

    # MERGE TABLES functionality
    @time_limit_exceeded_handler
    @broker_connection_closed_handler
    def get_merge_tables(self, context_id: str) -> List[str]:
        task_signature = self._celery_app.signature(TASK_SIGNATURES["get_merge_tables"])
        result = self._apply_async(
            task_signature=task_signature, context_id=context_id
        ).get(self._task_timeout)
        return result

    @time_limit_exceeded_handler
    @broker_connection_closed_handler
    def create_merge_table(
        self, context_id: str, command_id: str, table_names: List[str]
    ):
        task_signature = self._celery_app.signature(
            TASK_SIGNATURES["create_merge_table"]
        )
        result = self._apply_async(
            task_signature=task_signature,
            command_id=command_id,
            context_id=context_id,
            table_names=table_names,
        ).get(self._task_timeout)
        return result

    # REMOTE TABLES functionality
    @time_limit_exceeded_handler
    @broker_connection_closed_handler
    def get_remote_tables(self, context_id: str) -> List["TableInfo"]:
        task_signature = self._celery_app.signature(
            TASK_SIGNATURES["get_remote_tables"]
        )
        return self._apply_async(
            task_signature=task_signature, context_id=context_id
        ).get(self._task_timeout)

    @time_limit_exceeded_handler
    @broker_connection_closed_handler
    def create_remote_table(self, table_info: TableInfo, original_db_url: str) -> str:
        table_info_json = table_info.json()
        task_signature = self._celery_app.signature(
            TASK_SIGNATURES["create_remote_table"]
        )
        self._apply_async(
            task_signature=task_signature,
            table_info_json=table_info_json,
            monetdb_socket_address=original_db_url,
        ).get(self._task_timeout)

    # UDFs functionality
    def queue_run_udf(
        self,
        context_id: str,
        command_id: str,
        func_name: str,
        positional_args,
        keyword_args,
    ) -> QueuedUDFAsyncResult:
        task_signature = self._celery_app.signature(TASK_SIGNATURES["run_udf"])
        async_result = self._apply_async(
            task_signature=task_signature,
            command_id=command_id,
            context_id=context_id,
            func_name=func_name,
            positional_args_json=positional_args,
            keyword_args_json=keyword_args,
        )

        return QueuedUDFAsyncResult(
            node_id=self.node_id,
            command_id=command_id,
            context_id=context_id,
            func_name=func_name,
            positional_args=positional_args,
            keyword_args=keyword_args,
            async_result=async_result,
        )

    @time_limit_exceeded_handler
    @broker_connection_closed_handler
    def get_queued_udf_result(self, async_result: QueuedUDFAsyncResult):
        result = async_result.get(self._task_timeout)
        return result

    @time_limit_exceeded_handler
    @broker_connection_closed_handler
    def get_udfs(self, algorithm_name) -> List[str]:
        task_signature = self._celery_app.signature(TASK_SIGNATURES["get_udfs"])
        result = self._apply_async(
            task_signature=task_signature, algorithm_name=algorithm_name
        ).get(self._task_timeout)
        return result

    # return the generated monetdb pythonudf
    @time_limit_exceeded_handler
    @broker_connection_closed_handler
    def get_run_udf_query(
        self,
        context_id: str,
        command_id: str,
        func_name: str,
        positional_args: List[str],
    ) -> Tuple[str, str]:
        task_signature = self._celery_app.signature(
            TASK_SIGNATURES["get_run_udf_query"]
        )
        result = self._apply_async(
            task_signature=task_signature,
            command_id=command_id,
            context_id=context_id,
            func_name=func_name,
            positional_args_json=positional_args,
            keyword_args_json={},
        ).get(self._task_timeout)
        return result

    # CLEANUP functionality
    @broker_connection_closed_handler
    def clean_up(self, context_id: str):
        task_signature = self._celery_app.signature(TASK_SIGNATURES["clean_up"])
        self._apply_async(task_signature=task_signature, context_id=context_id).get(
            self._task_timeout
        )
