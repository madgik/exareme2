from typing import Union

import amqp
import billiard
import celery
import kombu
from celery import Celery
from celery.canvas import Signature
from celery.result import AsyncResult

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
        message = f"Timeout Exception: {timeout_type=} {connection_address=} while waiting for {async_result.task_id=}"
        super().__init__(message)


class CeleryWrapper:
    def __init__(self, socket_addr: str):
        self._socket_addr = socket_addr
        self._celery_app = self._create_new_celery_app()

    def _close(self):
        self._celery_app.close()

    # queue_task() is non-blocking, because apply_async() is non-blocking
    def queue_task(self, task_signature: str, *args, **kwargs) -> AsyncResult:
        try:
            task_signature = self._celery_app.signature(task_signature)
            async_result = task_signature.apply_async(args, kwargs)
            return async_result
        except (
            kombu.exceptions.OperationalError,
            amqp.exceptions.AccessRefused,
            amqp.exceptions.NotAllowed,
        ) as exc:
            # TODO how are things supposed to be logged in here?
            # get_backgroung_service_logger() makes no sense

            ######################
            print(f"(celery_app::CeleryWrapper::queue_task) {exc=}\n")

            import traceback

            tr = traceback.format_exc()
            print(tr)
            ######################

            self._close()
            self._celery_app = self._create_new_celery_app()

            connection_error = CeleryConnectionError(
                connection_address=self._socket_addr,
                error_details=f"while queuing {task_signature=} with {args=} and {kwargs=}",
            )

            raise connection_error

    # get_result() is blocking, because celery.result.AsyncResult.get() is blocking
    def get_result(
        self, async_result: AsyncResult, timeout: int
    ) -> Union[str, dict, list]:
        try:
            result = async_result.get(timeout)
            return result
        except (
            celery.exceptions.TimeoutError,
            billiard.exceptions.SoftTimeLimitExceeded,
            billiard.exceptions.TimeLimitExceeded,
        ) as timeout_error:
            # TODO how are things supposed to be logged in here?
            # get_backgroung_service_logger() makes no sense

            ######################
            print(
                f"(celery_app::CeleryWrapper::get_result) {timeout_error=} {async_result.id=}"
            )
            import traceback

            tr = traceback.format_exc()
            print(tr)
            ######################

            try:
                self._celery_app.control.inspect().ping()
            except kombu.exceptions.OperationalError as oper_err_inner:

                ######################
                print(
                    f"(celery_app::CeleryWrapper::get_result) {oper_err_inner=} {async_result.id=}"
                )

                import traceback

                tr = traceback.format_exc()
                print(tr)
                ######################

                self._close()
                self._celery_app = self._create_new_celery_app()

                connection_error = CeleryConnectionError(
                    connection_address=self._socket_addr,
                    error_details=f"while getting {async_result.id=}",
                )
                raise connection_error

            raise CeleryTaskTimeoutException(
                timeout_type=type(timeout_error),
                connection_address=self._socket_addr,
                async_result=async_result,
            )
        except kombu.exceptions.OperationalError as oper_err:
            # TODO how are things supposed to be logged in here?
            # get_backgroung_service_logger() makes no sense
            ######################
            print(f"(celery_app::CeleryWrapper::get_result) {oper_err=}")

            import traceback

            tr = traceback.format_exc()
            print(tr)
            ######################

            self._close()
            self._celery_app = self._create_new_celery_app()

            connection_error = CeleryConnectionError(
                connection_address=self._socket_addr,
                error_details=f"while getting {async_result.id=}",
            )
            raise connection_error

    def _create_new_celery_app(self):
        try:
            if self._celery_app:
                self._celery_app.close()
        except AttributeError:
            pass  # ignore it, there is no existing celery instance
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

    def reset(self):
        for _, celery_app in self._celery_apps.items():
            celery_app._close()
        self._celery_apps = {}
