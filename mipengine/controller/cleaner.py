import os
import threading
import time
import traceback
from datetime import datetime
from datetime import timezone
from pathlib import Path
from threading import RLock
from typing import List

import toml
from pydantic import BaseModel

from mipengine.controller import config as controller_config
from mipengine.controller import controller_logger as ctrl_logger
from mipengine.controller.algorithm_execution_tasks_handler import (
    NodeAlgorithmTasksHandler,
)
from mipengine.controller.node_landscape_aggregator import NodeLandscapeAggregator
from mipengine.singleton import Singleton

CLEANER_REQUEST_ID = "CLEANER"
CLEANUP_FILE_PREFIX = "cleanup"
CONTEXT_ID_CLEANUP_FILE = "contextids_cleanup.toml"

# Cleanup entry example:
# context_id= "3502300"
# node_ids = [ "testglobalnode", "testlocalnode1", "testlocalnode2",]
# timestamp = "2022-05-23T14:40:34.203085+00:00"
# released = true


class _NodeInfoDTO(BaseModel):
    node_id: str
    queue_address: str
    db_address: str
    tasks_timeout: int
    run_udf_task_timeout: int

    class Config:
        allow_mutation = False


class _CleanupEntry(BaseModel):
    context_id: str
    node_ids: List[str]
    timestamp: str
    released: bool


def _is_timestamp_expired(timestamp: str):
    now = datetime.now(timezone.utc)
    timestamp = datetime.fromisoformat(timestamp)
    time_elapsed = now - timestamp
    return time_elapsed.seconds > controller_config.cleanup.contextid_release_timelimit


class Cleaner(metaclass=Singleton):
    def __init__(self):
        self._logger = ctrl_logger.get_background_service_logger()

        self._cleanup_files_processor = CleanupFilesProcessor(self._logger)
        self._cleanup_interval = controller_config.cleanup.nodes_cleanup_interval

        self._keep_cleaning_up = True
        self._cleanup_loop_thread = None

    def _cleanup_loop(self):
        while self._keep_cleaning_up:
            try:
                all_entries = self._cleanup_files_processor.get_all_entries() or []
                for entry in all_entries:
                    if entry.released or _is_timestamp_expired(entry.timestamp):
                        failed_node_ids = self._exec_context_id_cleanup(
                            entry.context_id, entry.node_ids
                        )
                        if failed_node_ids:
                            self._logger.debug(
                                f"'Altering' file with {entry.context_id=}, removing node ids "
                                "wich succeded cleanup, keeping only failed node ids"
                            )
                            entry.node_ids = failed_node_ids
                            self._logger.debug(
                                f"Deleting file with {entry.context_id=}"
                            )
                            self._cleanup_files_processor.delete_file_by_context_id(
                                entry.context_id
                            )
                            self._logger.debug(
                                f"Creating file with {entry.context_id=}"
                            )
                            self._cleanup_files_processor.create_file_from_cleanup_entry(
                                entry
                            )

                        else:
                            self._logger.debug(
                                f"Cleanup for {entry.context_id=} complete. Deleting file"
                            )
                            self._cleanup_files_processor.delete_file_by_context_id(
                                entry.context_id
                            )

            except Exception as exc:
                self._logger.error(traceback.format_exc())

            finally:
                time.sleep(self._cleanup_interval)

    def _exec_context_id_cleanup(
        self, context_id: str, node_ids: [str]
    ) -> List[str]:  # returns failed node_ids
        node_task_handlers_to_async_results = {}
        for node_id in node_ids:
            try:
                node_info = self._get_node_info_by_id(node_id)
            except Exception as exc:
                self._logger.debug(
                    f"Could not get node info for {node_id=}. The node is "
                    "most likely offline"
                )
                continue
            task_handler = _get_node_task_handler(node_info)

            node_task_handlers_to_async_results[
                task_handler
            ] = task_handler.queue_cleanup(
                request_id=CLEANER_REQUEST_ID,
                context_id=context_id,
            )

        failed_node_ids = []
        for task_handler, async_result in node_task_handlers_to_async_results.items():
            try:
                task_handler.wait_queued_cleanup_complete(
                    async_result=async_result,
                    request_id=CLEANER_REQUEST_ID,
                )
            except Exception as exc:
                failed_node_ids.append(task_handler.node_id)
                self._logger.warning(
                    f"Cleanup task for {task_handler.node_id=}, for {context_id=} FAILED. "
                    f"Will retry in {self._cleanup_interval=} secs. Failure occured while "
                    "waiting the completion of the task (wait_queued_cleanup_complete), "
                    f"the exception raised was: {type(exc)}:{exc}"
                )
                continue

            self._logger.debug(
                f"Cleanup task succeeded for {task_handler.node_id=} for {context_id=}"
            )
        return failed_node_ids

    def start(self):
        self.stop()

        self._keep_cleaning_up = True
        self._cleanup_loop_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True
        )
        self._cleanup_loop_thread.start()

    def stop(self):
        if self._cleanup_loop_thread and self._cleanup_loop_thread.is_alive():
            self._keep_cleaning_up = False
            self._cleanup_loop_thread.join()

    def add_contextid_for_cleanup(
        self, context_id: str, algo_execution_node_ids: List[str]
    ):
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
        self._logger.debug(f"Setting released to true for file with {context_id=}")
        entry = self._cleanup_files_processor.get_entry_by_context_id(context_id)
        entry.released = True
        self._cleanup_files_processor.delete_file_by_context_id(context_id)  # by entry?
        self._cleanup_files_processor.create_file_from_cleanup_entry(entry)

    def _get_node_info_by_id(self, node_id: str) -> _NodeInfoDTO:
        node_info = NodeLandscapeAggregator().get_node_info(node_id)
        return _NodeInfoDTO(
            node_id=node_info.id,
            queue_address=":".join([str(node_info.ip), str(node_info.port)]),
            db_address=":".join([str(node_info.db_ip), str(node_info.db_port)]),
            tasks_timeout=controller_config.rabbitmq.celery_tasks_timeout,
            run_udf_task_timeout=controller_config.rabbitmq.celery_run_udf_task_timeout,
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
        tasks_timeout=2,  # node_info.tasks_timeout,
        run_udf_task_timeout=node_info.run_udf_task_timeout,
    )


class CleanupFilesProcessor:
    def __init__(self, logger):
        self._lock = RLock()
        self._logger = logger
        self._cleanup_entries_folder_path = Path(
            controller_config.cleanup.contextids_cleanup_folder
        )

        # Create the folder, if does not exist.
        self._cleanup_entries_folder_path.mkdir(parents=True, exist_ok=True)

    def create_file_from_cleanup_entry(self, entry: _CleanupEntry):
        entry_dict = entry.dict()
        filename = f"{CLEANUP_FILE_PREFIX}_{entry.context_id}.toml"
        full_path_and_filename = os.path.join(
            self._cleanup_entries_folder_path, filename
        )
        with open(full_path_and_filename, "x") as f:
            toml.dump(entry_dict, f)
        self._logger.debug(f"Created file with {entry.context_id=}")

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
            self._get_file_by_context_id(context_id).unlink()
        except FileNotFoundError as exc:
            self._logger.warning(
                f"Tried to delete {_file=} but file does not "
                f"exist. This should not happen. \n{exc=}"
            )
        self._logger.debug(f"Deleted file with {context_id=}")

    def get_entry_by_context_id(self, context_id) -> _CleanupEntry:
        _file = self._get_file_by_context_id(context_id)
        if _file:
            try:
                parsed_toml = toml.load(_file)
            except Exception as exc:
                self._logger.warning(
                    f"Trying to read {_file.name=} raised exception: {exc}"
                )
            return _CleanupEntry(**parsed_toml)
        else:
            self._logger.warning(
                f"Could not find entry and cleanup file with {context_id=}"
            )

    def _get_file_by_context_id(self, context_id: str):  # return type??
        for _file in self._cleanup_entries_folder_path.glob("cleanup_*.toml"):
            if context_id in _file.name:  # edge case
                try:
                    parsed_toml = toml.load(_file)
                except Exception as exc:
                    self._logger.warning(
                        f"Trying to read {_file.name=} raised exception: {exc}"
                    )
                if parsed_toml["context_id"] == context_id:
                    return _file
        self._logger.warning(f"Cleanup file with {context_id=} could not be found")

    def _delete_all_entries(self):
        for _file in self._cleanup_entries_folder_path.glob("cleanup_*.toml"):
            _file.unlink()
