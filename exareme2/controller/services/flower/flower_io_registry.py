import asyncio
from enum import Enum
from enum import unique
from typing import Any
from typing import Dict

from exareme2.controller.services.api.algorithm_request_dtos import (
    AlgorithmInputDataDTO,
)


@unique
class Status(str, Enum):
    SUCCESS = "success"
    RUNNING = "running"
    FAILURE = "failure"


class FlowerIORegistry:  # TODO OOO LOG EVERYTHING HERE!
    def __init__(self, logger):
        self._inputdata = None
        self._status = None
        self._result = None
        self.result_ready = None
        self._logger = logger
        self.reset_sync()
        # TODO OOO type annotations here

    def reset_sync(self):  # TODO OOO convert to private method, maybe delete?
        """Synchronously resets the algorithm execution info to initial state."""
        self._inputdata = AlgorithmInputDataDTO(data_model="", datasets=[])
        self._result = {}
        self._status = Status.RUNNING
        self.result_ready = asyncio.Event()
        self._logger.debug("Algorithm reset")

    async def reset(self):
        """Asynchronously resets the algorithm execution info to initial state."""
        self.reset_sync()

    async def set_result(
        self, result: Dict[str, Any]
    ):  # TODO OOO Result should use  a class with content and status keys
        """Sets the execution result and updates the status based on the presence of an error."""
        status = Status.FAILURE if "error" in result else Status.SUCCESS
        self._status = status
        self._result = result
        self._logger.debug("Result set with status: {}".format(status))
        self.result_ready.set()

    async def get_result(self):
        await self.result_ready.wait()
        return self._result

    async def get_result_with_timeout(self, timeout):
        try:
            # Wait for the result with a specified timeout
            await asyncio.wait_for(self.get_result(), timeout)
        except asyncio.TimeoutError:
            error = f"Failed to get result: operation timed out after {timeout} seconds"
            self._logger.error(error)
            self._result = {"error": error}
        return self._result

    def get_status(self) -> Status:
        """Returns the current status of the execution."""
        return self._status

    def set_inputdata(self, inputdata: AlgorithmInputDataDTO):
        """Sets new input data for the algorithm and resets status and result."""
        self._inputdata = inputdata
        self._status = Status.RUNNING
        self._result = {}
        self._logger.debug("Input data updated")

    def get_inputdata(self) -> AlgorithmInputDataDTO:
        """Returns the current input data."""
        return self._inputdata
