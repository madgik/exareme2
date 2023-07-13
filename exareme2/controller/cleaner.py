import os
import string
import threading
import time
import traceback
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import List

import toml
from pydantic import BaseModel

from exareme2.controller import controller_logger as ctrl_logger
from exareme2.controller.algorithm_execution_engine_tasks_handler import (
    NodeAlgorithmTasksHandler,
)
from exareme2.controller.node_landscape_aggregator import NodeLandscapeAggregator

CLEANER_REQUEST_ID = "CLEANER"
CLEANUP_FILE_TEMPLATE = string.Template("cleanup_${context_id}.toml")


class InitializationParams(BaseModel):
    cleanup_interval: int
    contextid_release_timelimit: int
    celery_cleanup_task_timeout: int
    celery_run_udf_task_timeout: int
    contextids_cleanup_folder: str
    node_landscape_aggregator: NodeLandscapeAggregator

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True


class _NodeInfoDTO(BaseModel):
    node_id: str
    queue_address: str
    db_address: str
    cleanup_task_timeout: int
    run_udf_task_timeout: int

    class Config:
        allow_mutation = False


class _CleanupEntry(BaseModel):
    context_id: str
    node_ids: List[str]
    timestamp: str
    released: bool


class Cleaner:
    """
    The Cleaner class handles the cleaning of database artifacts created during the
    execution of algorithms.

    How it works:
    Just before an algorithm starts executing, Cleaner::add_contextid_for_cleanup(context_id)
    is called (from the Controller). This creates a new file(ex. "cleanup_3502300.toml")
    containing a cleanup entry. As soon as the algorithm execution finishes,
    Cleaner::release_context_id(context_id) is called (from the Controller), which sets the
    'released' flag, of the respective cleanup entry, to 'true'. When the Cleaner object is
    started (method start()), it constantly loops through all the entries, finds the ones that
    either have their 'released' flag set to 'true' or their 'timestamp' has expired
    (check _is_timestamp_expired function) and processes them by calling the cleanup tasks
    on the respective nodes for the respective context_id. When the cleanup tasks on all
    the nodes of an entry are succesfull, the entry file is deleted. Otherwise the 'node_ids'
    list of the entry is updated to contain only the failed 'node_ids' and will be re-processed
    in the next iteration of the loop.

    Cleanup entry example:
    context_id= "3502300"
    node_ids = [ "testglobalnode", "testlocalnode1", "testlocalnode2",]
    timestamp = "2022-05-23T14:40:34.203085+00:00"
    released = false

    Parameters
    ----------
    init_params : InitializationParams
        A pydantic BaseModel that holds the initialization parameters
        of the Cleaner instance.

    Methods
    -------
    cleanup_context_id(context_id: str) -> bool:
        Execute the cleanup task on all nodes of a given context_id.
        Returns True if the cleanup task was successful on all nodes, False otherwise.

    start():
        Start the cleanup loop

    stop():
        Stop the cleanup loop

    add_contextid_for_cleanup(context_id: str, algo_execution_node_ids: List[str]):
        Create a new cleanup entry

    release_context_id(context_id):
        Set the "released" flag of the cleanup entry to true.
    """

    def __new__(cls, *args):
        if not hasattr(cls, "instance"):
            cls.instance = super(Cleaner, cls).__new__(cls)
            return cls.instance
        else:
            raise ValueError("Cleaner instance already exists.")

    @classmethod
    def _delete_instance(cls):
        if hasattr(cls, "instance"):
            del cls.instance

    def __init__(self, init_params: InitializationParams):
        """
        Initialize the Cleaner instance.

        Parameters
        ----------
        init_params : InitializationParams
            A pydantic BaseModel that holds the initialization parameters
            of the Cleaner instance.
        """
        self._logger = ctrl_logger.get_background_service_logger()

        self._cleanup_interval = init_params.cleanup_interval
        self._contextid_release_timelimit = init_params.contextid_release_timelimit
        self._celery_cleanup_task_timeout = init_params.celery_cleanup_task_timeout
        self._celery_run_udf_task_timeout = init_params.celery_run_udf_task_timeout
        self._contextids_cleanup_folder = init_params.contextids_cleanup_folder
        self._node_landscape_aggregator = init_params.node_landscape_aggregator
        self._cleanup_files_processor = CleanupFilesProcessor(
            self._logger, self._contextids_cleanup_folder
        )

        self._keep_cleaning_up = True
        self._cleanup_loop_thread = None

    def cleanup_context_id(self, context_id: str) -> bool:
        """
        Synchronously cleanup context_id. Calls the cleanup task on all the relevant
        nodes for the given context_id.

        Parameters
        ----------
        context_id : str
            The context_id of the finished algorithm execution that the cleanup task will be
            executed for.

        Returns
        -------
        bool
            True if the cleanup task was successful on all nodes, False otherwise.
        """
        # returns True if cleanup task was succesful for all nodes of the context_id
        entry = self._cleanup_files_processor.get_entry_by_context_id(context_id)
        return self._exec_cleanup(entry)

    def _cleanup_loop(self):
        while self._keep_cleaning_up:
            try:
                all_entries = self._cleanup_files_processor.get_all_entries() or []
                for entry in all_entries:
                    if entry.released or self._is_timestamp_expired(entry.timestamp):
                        self._exec_cleanup(entry)
            except Exception:
                self._logger.error(traceback.format_exc())
            finally:
                time.sleep(self._cleanup_interval)

    def _is_timestamp_expired(self, timestamp: str):
        now = datetime.now(timezone.utc)
        timestamp = datetime.fromisoformat(timestamp)
        time_elapsed = now - timestamp
        return time_elapsed.total_seconds() > self._contextid_release_timelimit

    def _exec_cleanup(self, entry: _CleanupEntry) -> bool:
        # returns True if cleanup task was succesful for all nodes of the context_id
        failed_node_ids = []
        node_task_handlers_to_async_results = {}
        for node_id in entry.node_ids:
            try:
                node_info = self._get_node_info_by_id(node_id)
            except Exception as exc:
                self._logger.debug(
                    f"Could not get node info for {node_id=}. The node is "
                    f"most likely offline exception:{exc=}"
                )
                failed_node_ids.append(node_id)
                continue
            task_handler = _get_node_task_handler(node_info)

            node_task_handlers_to_async_results[
                task_handler
            ] = task_handler.queue_cleanup(
                request_id=CLEANER_REQUEST_ID,
                context_id=entry.context_id,
            )

        for task_handler, async_result in node_task_handlers_to_async_results.items():
            try:
                task_handler.wait_queued_cleanup_complete(
                    async_result=async_result,
                    request_id=CLEANER_REQUEST_ID,
                )
            except Exception as exc:
                failed_node_ids.append(task_handler.node_id)
                self._logger.warning(
                    f"Cleanup task for {task_handler.node_id=}, for {entry.context_id=} FAILED. "
                    f"Will retry in {self._cleanup_interval=} secs. Failure occured while "
                    "waiting the completion of the task (wait_queued_cleanup_complete), "
                    f"the exception raised was: {type(exc)}:{exc}"
                )
                continue

            self._logger.debug(
                f"Cleanup task succeeded for {task_handler.node_id=} for {entry.context_id=}"
            )

        if not failed_node_ids:
            self._logger.debug(f"Cleanup for {entry.context_id=} complete.")
            self._cleanup_files_processor.delete_file_by_context_id(entry.context_id)
            return True
        else:
            self._logger.debug(
                f"'Altering' file with {entry.context_id=}, removing node ids "
                "which succeeded cleanup, keeping only failed node ids: "
                f"{failed_node_ids=}"
            )
            entry.node_ids = failed_node_ids
            self._cleanup_files_processor.update_file(entry)
            return False

    def start(self):
        """
        Start the cleanup loop
        """
        self.stop()

        self._keep_cleaning_up = True
        self._cleanup_loop_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True
        )
        self._cleanup_loop_thread.start()

    def stop(self):
        """
        Stop the cleanup loop
        """
        self._keep_cleaning_up = False
        if self._cleanup_loop_thread and self._cleanup_loop_thread.is_alive():
            self._cleanup_loop_thread.join()

    def add_contextid_for_cleanup(
        self, context_id: str, algo_execution_node_ids: List[str]
    ):

        """
        Create a new cleanup entry for the specified context_id along with the
        respective node_ids. Calling this method will not call the cleanup task on any
        node, it just "stores" the information that db artifacts with the specific
        context_id will be (potentially) created on the specified nodes

        Parameters
        ----------
        context_id : str
            The context_id of the algorithm execution.
        algo_execution_node_ids : List[str]
            The node_ids participating in the algorithm execution.
        """
        self._logger.debug(f"Creating file for new {context_id=}")
        now_timestamp = datetime.now(timezone.utc).isoformat()
        entry = _CleanupEntry(
            context_id=context_id,
            node_ids=algo_execution_node_ids,
            timestamp=now_timestamp,
            released=False,
        )
        self._cleanup_files_processor.create_file_from_cleanup_entry(entry)

    def release_context_id(self, context_id):
        """
        Asynchronously cleanup context_id. Sets the "released" flag of the cleanup entry
        to true and will call the cleanup task on the relevant nodes again and again in
        the "cleanup loop" until the task succeds on all nodes

        Parameters
        ----------
        context_id : str
            The context_id of the cleanup entry.
        """
        self._logger.debug(f"Setting released to true for file with {context_id=}")
        entry = self._cleanup_files_processor.get_entry_by_context_id(context_id)
        entry.released = True
        self._cleanup_files_processor.delete_file_by_context_id(context_id)
        self._cleanup_files_processor.create_file_from_cleanup_entry(entry)

    def _get_node_info_by_id(self, node_id: str) -> _NodeInfoDTO:
        node_info = self._node_landscape_aggregator.get_node_info(node_id)
        return _NodeInfoDTO(
            node_id=node_info.id,
            queue_address=":".join([str(node_info.ip), str(node_info.port)]),
            db_address=":".join([str(node_info.db_ip), str(node_info.db_port)]),
            cleanup_task_timeout=self._celery_cleanup_task_timeout,
            run_udf_task_timeout=self._celery_run_udf_task_timeout,
        )

    # This is only supposed to be called from a test.
    # In all other circumstances the cleaner should not be reset manually
    def _reset(self):
        self._cleanup_files_processor._delete_all_entries()


def _get_node_task_handler(node_info: _NodeInfoDTO) -> NodeAlgorithmTasksHandler:
    return NodeAlgorithmTasksHandler(
        node_id=node_info.node_id,
        node_queue_addr=node_info.queue_address,
        node_db_addr=node_info.db_address,
        tasks_timeout=node_info.cleanup_task_timeout,
        run_udf_task_timeout=node_info.run_udf_task_timeout,
    )


class CleanupFilesProcessor:
    def __init__(self, logger, contextids_cleanup_folder: str):
        self._logger = logger
        self._cleanup_entries_folder_path = Path(contextids_cleanup_folder)

        # Create the folder, if does not exist.
        self._cleanup_entries_folder_path.mkdir(parents=True, exist_ok=True)

    def create_file_from_cleanup_entry(self, entry: _CleanupEntry):
        entry_dict = entry.dict()
        filename = CLEANUP_FILE_TEMPLATE.substitute(context_id=entry.context_id)
        full_path_and_filename = os.path.join(
            self._cleanup_entries_folder_path, filename
        )
        with open(full_path_and_filename, "x") as f:
            toml.dump(entry_dict, f)
        self._logger.debug(f"Created file with {entry.context_id=}")

    def update_file(self, entry: _CleanupEntry):
        self._logger.debug(
            f"File with {entry.context_id=} is updated by deleting "
            "and re-creating it"
        )
        self.delete_file_by_context_id(entry.context_id)
        self.create_file_from_cleanup_entry(entry)

    def get_all_entries(self) -> [_CleanupEntry]:
        cleanup_entries = []
        for _file in self._cleanup_entries_folder_path.glob("cleanup_*.toml"):
            try:
                parsed_toml = toml.load(_file)
            except Exception as exc:
                self._logger.warning(
                    f"Trying to read {_file.name=} raised exception: {exc}"
                )

            cleanup_entries.append(_CleanupEntry(**parsed_toml))
        return cleanup_entries

    def delete_file_by_context_id(self, context_id: str):
        try:
            os.unlink(self._get_file_by_context_id(context_id))
        except FileNotFoundError as exc:
            self._logger.warning(
                f"Tried to delete file with {context_id=} but could not find such file. "
                f"This should not happen. \n{exc=}"
            )
        self._logger.debug(f"Deleted file with {context_id=}")

    def get_entry_by_context_id(self, context_id) -> _CleanupEntry:
        file_full_path = self._get_file_by_context_id(context_id)
        if file_full_path:
            try:
                parsed_toml = toml.load(file_full_path)
            except Exception as exc:
                self._logger.warning(
                    f"Trying to read {file_full_path=} raised exception: {exc}"
                )
                raise exc
            return _CleanupEntry(**parsed_toml)
        self._logger.warning(f"Could not find cleanup entry file with {context_id=}")

    def _get_file_by_context_id(self, context_id: str) -> str:
        filename_to_look_for = CLEANUP_FILE_TEMPLATE.substitute(context_id=context_id)
        file_full_path = os.path.join(
            self._cleanup_entries_folder_path, filename_to_look_for
        )
        try:
            parsed_toml = toml.load(file_full_path)
        except Exception as exc:
            self._logger.warning(
                f"Trying to read {file_full_path=} raised exception: {exc}"
            )
            return
        if parsed_toml["context_id"] != context_id:
            self._logger.error(f"File {file_full_path} contains wrnong {context_id=}")
        else:
            return file_full_path

    def _delete_all_entries(self):
        for _file in self._cleanup_entries_folder_path.glob("cleanup_*.toml"):
            _file.unlink()
