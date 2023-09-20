import enum
from typing import List
from typing import Optional

from pydantic import BaseModel


class SMPCRequestType(enum.Enum):
    INT_SUM = "sum"
    INT_MIN = "min"
    INT_MAX = "max"
    SUM = "fsum"
    MIN = "fmin"
    MAX = "fmax"

    def __str__(self):
        return self.name


class SMPCResponseStatus(enum.Enum):
    IN_QUEUE = "IN_QUEUE"
    RUNNING = "RUNNING"
    VALIDATING = "VALIDATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

    def __str__(self):
        return self.name


class DPRequestData(BaseModel):
    c: Optional[float]  # Differential Privacy - sensitivity
    e: Optional[float]  # Differential Privacy - privacy budget


class SMPCRequestData(BaseModel):
    computationType: SMPCRequestType
    clients: List[str]
    dp: Optional[DPRequestData]
    
class DifferentialPrivacyParams(BaseModel):
    sensitivity: float
    privacy_budget: float


class SMPCResponse(BaseModel):
    computationType: SMPCRequestType
    jobId: str
    status: SMPCResponseStatus


class SMPCResponseWithOutput(BaseModel):
    computationOutput: List[float]
    computationType: SMPCRequestType
    jobId: str
    status: SMPCResponseStatus
