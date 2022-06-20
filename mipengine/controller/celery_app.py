import time
import traceback
from threading import Lock
from threading import Thread
from typing import Union

import amqp
import billiard
import celery
import kombu
from celery import Celery
from celery.canvas import Signature
from celery.result import AsyncResult

from mipengine.controller import config as controller_config
from mipengine.controller import controller_logger
from mipengine.singleton import Singleton


class Logger:
    def info(msg: str, *args, **kwargs):
        pass

    def debug(msg: str, *args, **kwargs):
        pass

    def error(msg: str, *args, **kwargs):
        pass


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
        self._instantiate_new_celery_object_lock = Lock()
        self._instantiate_new_celery_object()

    def _close(self):
        self._celery_app.close()

    # queue_task() is non-blocking, because apply_async() is non-blocking
    def queue_task(
        self, task_signature: str, logger: Logger, *args, **kwargs
    ) -> AsyncResult:
        try:
            task_signature = self._celery_app.signature(task_signature)
            async_result = task_signature.apply_async(args, kwargs)
            return async_result
        except (
            kombu.exceptions.OperationalError,
            amqp.exceptions.AccessRefused,
            amqp.exceptions.NotAllowed,
        ) as exc:
            tr = traceback.format_exc()
            logger.error(tr)

            if not self._instantiate_new_celery_object_lock.locked():
                self._start_instantiate_new_celery_object_thread()

            connection_error = CeleryConnectionError(
                connection_address=self._socket_addr,
                error_details=f"while queuing {task_signature=} with {args=} and {kwargs=}",
            )

            raise connection_error

    # get_result() is blocking, because celery.result.AsyncResult.get() is blocking
    def get_result(
        self, async_result: AsyncResult, timeout: int, logger: Logger
    ) -> Union[str, dict, list]:
        try:
            result = async_result.get(timeout)
            return result

        except (
            celery.exceptions.TimeoutError,
            billiard.exceptions.SoftTimeLimitExceeded,
            billiard.exceptions.TimeLimitExceeded,
        ) as timeout_error:

            tr = traceback.format_exc()
            logger.error(tr)

            try:
                self._celery_app.control.inspect().ping()
            except kombu.exceptions.OperationalError as oper_err_inner:
                tr = traceback.format_exc()
                logger.error(tr)

                if not self._instantiate_new_celery_object_lock.locked():
                    self._start_instantiate_new_celery_object_thread()

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
            tr = traceback.format_exc()
            logger.error(tr)

            if not self._instantiate_new_celery_object_lock.locked():
                self._start_instantiate_new_celery_object_thread()

            connection_error = CeleryConnectionError(
                connection_address=self._socket_addr,
                error_details=f"while getting {async_result.id=}",
            )
            raise connection_error

    def _start_instantiate_new_celery_object_thread(self):
        instantiate_new_celery_object_thread = Thread(
            target=self._instantiate_new_celery_object, daemon=True
        )
        instantiate_new_celery_object_thread.start()

    def _instantiate_new_celery_object(self):
        with self._instantiate_new_celery_object_lock:
            logger = controller_logger.get_background_service_logger()
            logger.info(
                f"Instantiating new Celery object for CeleryWrapper with {self._socket_addr=}"
            )
            print(f"")

            if hasattr(self, "_celery_app"):
                self._celery_app.close()
            user = controller_config.rabbitmq.user
            password = controller_config.rabbitmq.password
            vhost = controller_config.rabbitmq.vhost
            broker = f"pyamqp://{user}:{password}@{self._socket_addr}/{vhost}"
            celery_app = Celery(broker=broker, backend="rpc://")

            # connection pool disabled
            # connections are established and closed for every use
            celery_app.conf.broker_pool_limit = None

            # check connection
            connection_is_ok = False
            retry_interval = 1
            while not connection_is_ok:
                try:
                    celery_app.control.inspect().ping()
                    connection_is_ok = True
                except kombu.exceptions.OperationalError:
                    logger.debug(
                        f"Connection to broker ({self._socket_addr=}) is not established. Will retry in {retry_interval} seconds."
                    )
                    celery_app = Celery(broker=broker, backend="rpc://")
                    celery_app.conf.broker_pool_limit = None

                time.sleep(retry_interval)
            logger.info(
                f"Connection to broker ({self._socket_addr=}) successfully established."
            )
            self._celery_app = celery_app


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
        for celery_app in self._celery_apps.values():
            celery_app._close()
        self._celery_apps = {}
