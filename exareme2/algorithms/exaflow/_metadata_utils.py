from __future__ import annotations

from collections.abc import Mapping
from typing import List
from typing import Optional


def get_metadata_categories(
    metadata,
    variable: str,
    *,
    min_length: Optional[int] = None,
    expected_length: Optional[int] = None,
    context: Optional[str] = None,
) -> List[str]:
    context_hint = f" for {context}" if context else ""

    if isinstance(metadata, Mapping):
        variable_metadata = metadata.get(variable)
    else:
        variable_metadata = getattr(metadata, variable, None)

    if variable_metadata is None:
        raise ValueError(f"Metadata{context_hint} for variable '{variable}' not found.")

    if isinstance(variable_metadata, Mapping):
        enumerations = variable_metadata.get("enumerations")
    else:
        enumerations = getattr(variable_metadata, "enumerations", None)

    if not enumerations:
        raise ValueError(
            f"Variable '{variable}' must have enumerations defined in metadata{context_hint}."
        )

    if isinstance(enumerations, Mapping):
        categories = list(enumerations.keys())
    else:
        categories = list(enumerations)

    if not categories:
        raise ValueError(
            f"Enumerations for variable '{variable}' are empty{context_hint}."
        )

    if expected_length is not None and len(categories) != expected_length:
        raise ValueError(
            f"Variable '{variable}' must have exactly {expected_length} categories{context_hint}, "
            f"got {len(categories)}."
        )

    if min_length is not None and len(categories) < min_length:
        raise ValueError(
            f"Variable '{variable}' must have at least {min_length} categories{context_hint}, "
            f"got {len(categories)}."
        )

    return categories
