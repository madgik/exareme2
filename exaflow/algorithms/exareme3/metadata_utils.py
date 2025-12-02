from typing import Any
from typing import Dict
from typing import Iterable
from typing import TypedDict


class VariableMetadata(TypedDict, total=False):
    """
    Minimal metadata schema required by the exaflow algorithms.

    This mirrors the implicit contract in the migrated flows: each variable
    needs an `is_categorical` flag so flows can split variables into
    categorical vs numerical groups for encoding. Some algorithms also expect
    `enumerations` to be present for categorical variables.
    """

    is_categorical: bool
    enumerations: Dict[str, Any]


def validate_metadata_vars(
    required_vars: Iterable[str], metadata: Dict[str, VariableMetadata]
) -> None:
    """
    Validate that metadata contains the required variables with the expected keys.

    Raises
    ------
    KeyError
        If a required variable is missing from metadata.
    KeyError
        If the `is_categorical` flag is missing for a required variable.
    """

    for var in required_vars:
        if var not in metadata:
            raise KeyError(f"Metadata missing required variable: {var!r}")
        if "is_categorical" not in metadata[var]:
            raise KeyError(
                f"Metadata for variable {var!r} must include 'is_categorical'"
            )


def validate_metadata_enumerations(
    required_vars: Iterable[str], metadata: Dict[str, VariableMetadata]
) -> None:
    """
    Validate that metadata contains enumerations for the required variables.

    Raises
    ------
    KeyError
        If a required variable is missing from metadata.
    KeyError
        If the `enumerations` entry is missing for a required variable.
    """

    for var in required_vars:
        if var not in metadata:
            raise KeyError(f"Metadata missing required variable: {var!r}")
        if "enumerations" not in metadata[var]:
            raise KeyError(f"Metadata for variable {var!r} must include 'enumerations'")
