import glob
import importlib
import os
from os.path import basename
from os.path import isfile
from types import ModuleType
from typing import Dict

from exareme2.algorithms.in_database.algorithm import Algorithm
from exareme2.algorithms.in_database.algorithm import AlgorithmDataLoader
from exareme2.datatypes import DType
from exareme2.utils import AttrDict

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
ALGORITHM_FOLDERS = (
    "./exareme2/algorithms/in_database,./exareme2/algorithms/native_python"
)
if algorithm_folders := os.getenv(ALGORITHM_FOLDERS_ENV_VARIABLE):
    ALGORITHM_FOLDERS = algorithm_folders


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


import_algorithm_modules()


def get_algorithm_classes() -> Dict[str, type]:
    return {cls.algname: cls for cls in Algorithm.__subclasses__()}


def get_algorithm_data_loaders() -> Dict[str, type]:
    return {cls.algname: cls for cls in AlgorithmDataLoader.__subclasses__()}


def _check_algo_naming_matching(algo_classes: dict, algo_data_loaders: dict):
    algo_classes_set = set(algo_classes.keys())
    algo_data_loaders_set = set(algo_data_loaders.keys())
    sym_diff = algo_classes_set.symmetric_difference(algo_data_loaders_set)
    if sym_diff:
        raise AlgorithmNamesMismatchError(sym_diff, algo_classes, algo_data_loaders)


algorithm_classes = get_algorithm_classes()
algorithm_data_loaders = get_algorithm_data_loaders()
_check_algo_naming_matching(
    algo_classes=algorithm_classes, algo_data_loaders=algorithm_data_loaders
)
