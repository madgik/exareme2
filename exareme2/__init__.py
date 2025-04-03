import glob
import importlib
import os
from os.path import basename
from os.path import isfile
from types import ModuleType
from typing import Dict

from exareme2.algorithms.exaflow.algorithm import Algorithm as ExaflowAlgorithm
from exareme2.algorithms.exareme2.algorithm import Algorithm as Exareme2Algorithm
from exareme2.algorithms.exareme2.algorithm import AlgorithmDataLoader
from exareme2.datatypes import DType
from exareme2.utils import AttrDict

__all__ = [
    "DType",
    "AttrDict",
    "EXAREME2_ALGORITHM_FOLDERS_ENV_VARIABLE",
    "EXAREME2_ALGORITHM_FOLDERS",
    "exareme2_algorithm_classes",
    "DATA_TABLE_PRIMARY_KEY",
    "flower_algorithm_folder_paths",
    "FLOWER_ALGORITHM_FOLDERS_ENV_VARIABLE",
    "FLOWER_ALGORITHM_FOLDERS",
    "EXAFLOW_ALGORITHM_FOLDERS_ENV_VARIABLE",
    "EXAFLOW_ALGORITHM_FOLDERS",
    "exaflow_algorithm_classes",
]

DATA_TABLE_PRIMARY_KEY = "row_id"

EXAREME2_ALGORITHM_FOLDERS_ENV_VARIABLE = "EXAREME2_ALGORITHM_FOLDERS"
EXAREME2_ALGORITHM_FOLDERS = os.getenv(
    EXAREME2_ALGORITHM_FOLDERS_ENV_VARIABLE, "./exareme2/algorithms/exareme2"
)


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


def import_algorithm_modules(algorithm_folders: str) -> Dict[str, ModuleType]:
    """
    Import all algorithm modules from the given folder paths.

    :param algorithm_folders: Comma-separated string of folder paths.
    :return: A dictionary mapping module names to imported module objects.
    """
    all_modules = {}
    for algorithm_folder in algorithm_folders.split(","):
        # Get all .py files in the folder (excluding __init__.py)
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
        for spec, module in zip(specs, modules.values()):
            spec.loader.exec_module(module)
        all_modules.update(modules)
    return all_modules


def get_exareme2_algorithm_classes() -> Dict[str, type]:
    import_algorithm_modules(EXAREME2_ALGORITHM_FOLDERS)
    return {cls.algname: cls for cls in Exareme2Algorithm.__subclasses__()}


def get_exareme2_algorithm_data_loaders() -> Dict[str, type]:
    return {cls.algname: cls for cls in AlgorithmDataLoader.__subclasses__()}


def _check_algo_naming_matching(algo_classes: dict, algo_data_loaders: dict):
    algo_classes_set = set(algo_classes.keys())
    algo_data_loaders_set = set(algo_data_loaders.keys())
    sym_diff = algo_classes_set.symmetric_difference(algo_data_loaders_set)
    if sym_diff:
        raise AlgorithmNamesMismatchError(sym_diff, algo_classes, algo_data_loaders)


exareme2_algorithm_classes = get_exareme2_algorithm_classes()
exareme2_algorithm_data_loaders = get_exareme2_algorithm_data_loaders()
_check_algo_naming_matching(
    algo_classes=exareme2_algorithm_classes,
    algo_data_loaders=exareme2_algorithm_data_loaders,
)


def find_flower_algorithm_folder_paths(algorithm_folders):
    # Split the input string into a list of folder paths
    folder_paths = algorithm_folders.split(",")

    # Initialize an empty dictionary to store the result
    algorithm_folder_paths = {}

    # Iterate over each folder path
    for folder_path in folder_paths:
        if not os.path.isdir(folder_path):
            continue  # Skip if the path is not a valid directory

        # List all files and folders in the current folder path
        items = os.listdir(folder_path)

        # Filter for .json files and corresponding folders
        for item in items:
            if item.endswith(".json"):
                algorithm_name = item[:-5]  # Remove '.json' to get the algorithm name
                algorithm_folder = os.path.join(folder_path, algorithm_name)
                if os.path.isdir(algorithm_folder):
                    # Store the algorithm name and the complete folder path in the dictionary
                    algorithm_folder_paths[algorithm_name] = algorithm_folder

    return algorithm_folder_paths


FLOWER_ALGORITHM_FOLDERS_ENV_VARIABLE = "FLOWER_ALGORITHM_FOLDERS"
FLOWER_ALGORITHM_FOLDERS = "./exareme2/algorithms/flower"
if flower_algorithm_folders := os.getenv(FLOWER_ALGORITHM_FOLDERS_ENV_VARIABLE):
    FLOWER_ALGORITHM_FOLDERS = flower_algorithm_folders

flower_algorithm_folder_paths = find_flower_algorithm_folder_paths(
    FLOWER_ALGORITHM_FOLDERS
)


EXAFLOW_ALGORITHM_FOLDERS_ENV_VARIABLE = "EXAFLOW_ALGORITHM_FOLDERS"
EXAFLOW_ALGORITHM_FOLDERS = os.getenv(
    EXAFLOW_ALGORITHM_FOLDERS_ENV_VARIABLE, "./exareme2/algorithms/exaflow"
)


def get_exaflow_algorithm_classes() -> Dict[str, type]:
    import_algorithm_modules(EXAFLOW_ALGORITHM_FOLDERS)
    return {cls.algname: cls for cls in ExaflowAlgorithm.__subclasses__()}


exaflow_algorithm_classes = get_exaflow_algorithm_classes()
