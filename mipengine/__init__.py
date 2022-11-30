import glob
import importlib
import os
from importlib import util
from os.path import basename
from os.path import isfile
from types import ModuleType
from typing import Dict

from mipengine.algorithms.algorithm import Algorithm

from .attrdict import AttrDict
from .datatypes import DType

__all__ = [
    "DType",
    "AttrDict",
    "ALGORITHM_FOLDERS_ENV_VARIABLE",
    "ALGORITHM_FOLDERS",
    "algorithm_modules",
    "DATA_TABLE_PRIMARY_KEY",
]

DATA_TABLE_PRIMARY_KEY = "row_id"

ALGORITHM_FOLDERS_ENV_VARIABLE = "ALGORITHM_FOLDERS"
ALGORITHM_FOLDERS = "./mipengine/algorithms"
if algorithm_folders := os.getenv(ALGORITHM_FOLDERS_ENV_VARIABLE):
    ALGORITHM_FOLDERS = algorithm_folders


def import_algorithm_modules() -> Dict[str, ModuleType]:
    # Import all algorithm modules
    # Import all .py modules in the algorithm folder paths
    # https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path?page=1&tab=votes#tab-top

    all_modules = {}
    for algorithm_folder in ALGORITHM_FOLDERS.split(","):
        all_module_paths = glob.glob(f"{algorithm_folder}/*.py")
        algorithm_module_paths = [
            module
            for module in all_module_paths
            if isfile(module) and not module.endswith("__init__.py")
        ]
        algorithm_names = [
            basename(module_path)[:-3] for module_path in algorithm_module_paths
        ]
        specs = [
            importlib.util.spec_from_file_location(algorithm_name, module_path)
            for algorithm_name, module_path in zip(
                algorithm_names, algorithm_module_paths
            )
        ]
        modules = {
            name: importlib.util.module_from_spec(spec)
            for name, spec in zip(algorithm_names, specs)
        }

        # Import modules
        [
            spec.loader.exec_module(module)
            for spec, module in zip(specs, modules.values())
        ]

        all_modules.update(modules)

    return all_modules


def import_algorithm_classes() -> Dict[str, type]:
    import_algorithm_modules()
    return {cls.algname: cls for cls in Algorithm.__subclasses__()}


algorithm_classes = import_algorithm_classes()
