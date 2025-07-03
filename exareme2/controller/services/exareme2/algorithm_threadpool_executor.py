import asyncio
import concurrent
import traceback

from exareme2.controller.celery.app import CeleryConnectionError
from exareme2.controller.celery.app import CeleryTaskTimeoutException
from exareme2.controller.services.errors import WorkerTaskTimeoutError
from exareme2.controller.services.errors import WorkerUnresponsiveError

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
        raise WorkerUnresponsiveError()
    except CeleryTaskTimeoutException as exc:
        logger.error(f"ErrorType: '{type(exc)}' and message: '{exc}'")
        raise WorkerTaskTimeoutError()
    except Exception as exc:
        logger.error(traceback.format_exc())
        raise exc
