import glob
import importlib
import os
from importlib import util
from os.path import basename
from os.path import isfile
from types import ModuleType
from typing import Dict

from mipengine.algorithms.algorithm import Algorithm
from mipengine.algorithms.algorithm import AlgorithmDataLoader

from .attrdict import AttrDict
from .datatypes import DType

__all__ = [
    "DType",
    "AttrDict",
    "ALGORITHM_FOLDERS_ENV_VARIABLE",
    "ALGORITHM_FOLDERS",
    "algorithm_classes",
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


def get_algorithm_classes() -> Dict[str, type]:
    import_algorithm_modules()
    return {cls.algname: cls for cls in Algorithm.__subclasses__()}


def get_algorithms_data_loader() -> Dict[str, type]:
    import_algorithm_modules()
    return {cls.algname: cls for cls in AlgorithmDataLoader.__subclasses__()}


algorithm_classes = get_algorithm_classes()
algorithm_data_loaders = get_algorithms_data_loader()


class AlgorithmNamesMismatchError(Exception):
    def __init__(self, mismatches, algorithm_classes, algorithm_data_loaders):
        mismatches_algname_class = []
        for m in mismatches:
            alg_classes = algorithm_classes.get(m)
            alg_dloaders = algorithm_data_loaders.get(m)
            if alg_classes:
                mismatches_algname_class.append(f"{m} -> {alg_classes}")
            elif alg_dloaders:
                mismatches_algname_class.append(f"{m} -> {alg_dloaders}")
        message = (
            "The following Algorithm and AlgorithmDataLoader classes have "
            f"mismatching 'algname' values: {mismatches_algname_class}"
        )
        super().__init__(message)
        self.message = message


alg_classes_set = set(algorithm_classes.keys())
alg_dloaders_set = set(algorithm_data_loaders.keys())
sym_diff = alg_classes_set.symmetric_difference(alg_dloaders_set)
if sym_diff:
    raise AlgorithmNamesMismatchError(
        sym_diff, algorithm_classes, algorithm_data_loaders
    )
