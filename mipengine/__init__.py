import os

from .datatypes import DType
from .attrdict import AttrDict

__all__ = ["DType", "AttrDict", "ALGORITHMS_FOLDER_ENV_VARIABLE", "ALGORITHMS_FOLDER"]

ALGORITHMS_FOLDER_ENV_VARIABLE = "ALGORITHMS_FOLDER"
ALGORITHMS_FOLDER = "mipengine.algorithms"

if os.getenv(ALGORITHMS_FOLDER_ENV_VARIABLE):
    ALGORITHMS_FOLDER = os.getenv(ALGORITHMS_FOLDER_ENV_VARIABLE)
