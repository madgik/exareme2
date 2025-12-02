from typing import List
from typing import Optional

from exaflow.worker_communication import BadUserInput


def require_dependent_var(inputdata, *, message: Optional[str] = None) -> List[str]:
    """
    Ensure inputdata.y exists and is non-empty. Returns the list for convenience.
    """
    if not inputdata.y:
        raise BadUserInput(message or "A dependent variable is required.")
    return inputdata.y


def require_covariates(inputdata, *, message: Optional[str] = None) -> List[str]:
    """
    Ensure inputdata.x exists and is non-empty. Returns the list for convenience.
    """
    if not inputdata.x:
        raise BadUserInput(message or "At least one covariate is required.")
    return inputdata.x


def require_exact_dependents(
    inputdata, count: int, *, message: Optional[str] = None
) -> List[str]:
    """
    Ensure inputdata.y contains exactly `count` elements. Returns the list.
    """
    ys = require_dependent_var(inputdata, message=message)
    if len(ys) != count:
        raise BadUserInput(
            message or f"Exactly {count} dependent variable(s) required; got {len(ys)}."
        )
    return ys


def require_exact_covariates(
    inputdata, count: int, *, message: Optional[str] = None
) -> List[str]:
    """
    Ensure inputdata.x contains exactly `count` elements. Returns the list.
    """
    xs = require_covariates(inputdata, message=message)
    if len(xs) != count:
        raise BadUserInput(
            message or f"Exactly {count} covariate(s) required; got {len(xs)}."
        )
    return xs
