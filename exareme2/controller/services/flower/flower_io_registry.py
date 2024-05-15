import asyncio
from enum import Enum
from enum import unique
from typing import Any
from typing import Dict
from typing import Optional

from exareme2.controller.services.api.algorithm_request_dtos import (
    AlgorithmInputDataDTO,
)


@unique
class Status(str, Enum):
    SUCCESS = "success"
    RUNNING = "running"
    FAILURE = "failure"


class Result:
    def __init__(self, content: Dict[str, Any], status: Status):
        self.content = content
        self.status = status

    def __repr__(self):
        return f"Result(status={self.status}, content={self.content})"


class FlowerIORegistry:
    def __init__(self, timeout, logger):
        self._inputdata: Optional[AlgorithmInputDataDTO] = None
        self._result: Optional[Result] = None
        self.result_ready: Optional[asyncio.Event] = None
        self._logger = logger
        self._reset_sync()
        self._timeout = timeout

    def _reset_sync(self):
        """Synchronously resets the algorithm execution info to initial state."""
        self._inputdata = AlgorithmInputDataDTO(data_model="", datasets=[])
        self._result = Result(content={}, status=Status.RUNNING)
        self.result_ready = asyncio.Event()
        self._logger.debug("Algorithm reset: input data cleared, status set to RUNNING")

    async def reset(self):
        """Asynchronously resets the algorithm execution info to initial state."""
        self._reset_sync()
        self._logger.debug("Asynchronous reset complete")

    async def set_result(self, result: Dict[str, Any]):
        """Sets the execution result and updates the status based on the presence of an error."""
        status = Status.FAILURE if "error" in result else Status.SUCCESS
        self._result = Result(content=result, status=status)
        self._logger.debug(f"Result set with status: {status}, content: {result}")
        self.result_ready.set()

    async def get_result(self) -> Dict[str, Any]:
        await self.result_ready.wait()
        self._logger.debug(f"Result retrieved: {self._result}")
        return self._result.content

    async def get_result_with_timeout(self) -> Dict[str, Any]:
        try:
            await asyncio.wait_for(self.get_result(), self._timeout)
        except asyncio.TimeoutError:
            error = f"Failed to get result: operation timed out after {self._timeout} seconds"
            self._logger.error(error)
            self._result = Result(content={"error": error}, status=Status.FAILURE)
        self._logger.debug(f"Result with timeout: {self._result}")
        return self._result.content

    def get_status(self) -> Status:
        """Returns the current status of the execution."""
        status = self._result.status if self._result else Status.RUNNING
        self._logger.debug(f"Status retrieved: {status}")
        return status

    def set_inputdata(self, inputdata: AlgorithmInputDataDTO):
        """Sets new input data for the algorithm and resets status and result."""
        self._inputdata = inputdata
        self._result = Result(content={}, status=Status.RUNNING)
        self.result_ready.clear()
        self._logger.debug(f"Input data updated: {inputdata}, status reset to RUNNING")

    def get_inputdata(self) -> AlgorithmInputDataDTO:
        """Returns the current input data."""
        self._logger.debug(f"Input data retrieved: {self._inputdata}")
        return self._inputdata
