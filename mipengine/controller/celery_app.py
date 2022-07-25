import traceback
from logging import Logger
from threading import Lock

from amqp import AccessRefused
from amqp import NotAllowed
from amqp import UnexpectedFrame
from celery import Celery
from celery import exceptions as celery_exceptions
from celery.result import AsyncResult
from kombu import exceptions as kombu_exceptions

from mipengine.controller import config as controller_config
from mipengine.singleton import Singleton


class CeleryConnectionError(Exception):
    def __init__(self, connection_address: str, error_details: str):
        message = f"Connection Error: {connection_address=} {error_details=}"
        super().__init__(message)


class CeleryTaskTimeoutException(Exception):
    def __init__(
        self, timeout_type: str, connection_address: str, async_result: AsyncResult
    ):
        message = (
            f"Timeout Exception: {timeout_type=} {connection_address=} while waiting "
            f"for {async_result.task_id=}"
        )
        super().__init__(message)


class CeleryWrapper:
    def __init__(self, socket_addr: str):
        self._socket_addr = socket_addr
        self._change_app_lock = Lock()
        self._celery_app = self._instantiate_celery_object()

    def _set_new_cel_app(self, cel_app: Celery):
        with self._change_app_lock:
            old_app = self._celery_app
            self._celery_app = cel_app
            old_app.close()

    # queue_task() is non-blocking, because apply_async() is non-blocking
    def queue_task(
        self, task_signature: str, logger: Logger, *args, **kwargs
    ) -> AsyncResult:
        try:
            task_signature = self._celery_app.signature(task_signature)
            return task_signature.apply_async(args, kwargs)
        except (
            kombu_exceptions.OperationalError,
            AccessRefused,
            NotAllowed,
        ) as exc:
            logger.error(
                f"{self._socket_addr=} Exception: {exc=} was caught. Most likely this "
                f"means the broker is not accessible or it's being started. Queuing of {task_signature=} with "
                f"{args=} and {kwargs=} FAILED."
            )
            raise CeleryConnectionError(
                connection_address=self._socket_addr,
                error_details=f"While queuing {task_signature=} with {args=} and {kwargs=} on {self._socket_addr=}.",
            )
        except Exception as exc:
            logger.error(traceback.format_exc())
            raise exc

    # get_result() is blocking, because celery.result.AsyncResult.get() is blocking
    def get_result(self, async_result: AsyncResult, timeout: int, logger: Logger):
        try:
            return async_result.get(timeout)
        except (
            OSError,
            UnexpectedFrame,
            ConnectionResetError,
            ConnectionRefusedError,
            kombu_exceptions.OperationalError,
        ) as exc:
            logger.error(
                f"{self._socket_addr=} Exception: {exc=} was raised. Most likely this "
                f"means that the rabbitmq is down."
            )

            # The celery app needs to be recreated due to a bug:
            # https://github.com/celery/celery/issues/6912#issuecomment-1107260087
            # Create a new celery app since the current one is corrupted.
            self._set_new_cel_app(self._instantiate_celery_object())

            raise CeleryConnectionError(
                connection_address=self._socket_addr,
                error_details=f"While getting {async_result.id=}.",
            )
        except celery_exceptions.TimeoutError as timeout_error:
            raise CeleryTaskTimeoutException(
                timeout_type=str(type(timeout_error)),
                connection_address=self._socket_addr,
                async_result=async_result,
            )
        except Exception as exc:
            logger.error(traceback.format_exc())
            raise exc

    # It seems that Celery objects are somewhat expensive, do not have more than one
    # instance per node at any time
    def _instantiate_celery_object(self) -> Celery:
        user = controller_config.rabbitmq.user
        password = controller_config.rabbitmq.password
        vhost = controller_config.rabbitmq.vhost
        broker = f"pyamqp://{user}:{password}@{self._socket_addr}/{vhost}"
        celery_app = Celery(broker=broker, backend="rpc://")

        # connection pool disabled
        # connections are established and closed for every use
        celery_app.conf.broker_pool_limit = None
        return celery_app


class CeleryAppFactory(metaclass=Singleton):
    def __init__(self):
        self._celery_apps = {}

    def get_celery_app(self, socket_addr: str) -> CeleryWrapper:
        if socket_addr in self._celery_apps:
            return self._celery_apps[socket_addr]
        else:
            self._celery_apps[socket_addr] = CeleryWrapper(socket_addr)
            return self._celery_apps[socket_addr]
