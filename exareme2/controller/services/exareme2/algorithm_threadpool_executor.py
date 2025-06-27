import asyncio
import concurrent
import traceback

from exareme2.controller.celery.app import CeleryConnectionError
from exareme2.controller.celery.app import CeleryTaskTimeoutException

_thread_pool_executor = concurrent.futures.ThreadPoolExecutor()


# TODO add types
# TODO change func name, Transformers runs through this as well
async def algorithm_run_in_threadpool(algorithm, data_model_views, metadata, logger):
    # By calling blocking method Algorithm.run() inside run_in_executor(),
    # Algorithm.run() will execute in a separate thread of the threadpool and at
    # the same time yield control to the executor event loop, through await

    try:
        loop = asyncio.get_event_loop()
        algorithm_result = await loop.run_in_executor(
            _thread_pool_executor,
            algorithm.run,
            data_model_views.to_list(),
            metadata,
        )
        return algorithm_result
    except CeleryConnectionError as exc:
        logger.error(f"ErrorType: '{type(exc)}' and message: '{exc}'")
        raise WorkerUnresponsiveException()
    except CeleryTaskTimeoutException as exc:
        logger.error(f"ErrorType: '{type(exc)}' and message: '{exc}'")
        raise WorkerTaskTimeoutException()
    except Exception as exc:
        logger.error(traceback.format_exc())
        raise exc


class WorkerUnresponsiveException(Exception):
    def __init__(self):
        message = (
            "One of the workers participating in the algorithm execution "
            "stopped responding"
        )
        super().__init__(message)
        self.message = message


class WorkerTaskTimeoutException(Exception):
    def __init__(self):
        message = (
            f"One of the celery in the algorithm execution took longer to finish than "
            f"the timeout.This could be caused by a high load or by an experiment with "
            f"too much data. Please try again or increase the timeout."
        )
        super().__init__(message)
        self.message = message
