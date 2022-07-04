import os
import threading
import time
from datetime import datetime
from datetime import timezone
from pathlib import Path
from threading import Lock
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
CONTEXT_ID_CLEANUP_FILE = "contextids_cleanup.toml"

# Cleanup entry example:
# [3502300] <--context_id
# nodes = [ "testglobalnode", "testlocalnode1", "testlocalnodetmp",]
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


class Cleaner(metaclass=Singleton):
    def __init__(self):
        self._logger = ctrl_logger.get_background_service_logger()

        self._cleanup_file_processor = CleanupFileProcessor(self._logger)
        self._clean_up_interval = controller_config.cleanup.nodes_cleanup_interval

        self._keep_cleaning_up = True
        self._cleanup_loop_thread = None

    def _cleanup_loop(self):
        while self._keep_cleaning_up:
            try:
                contextids_and_status = self._cleanup_file_processor.read_cleanup_file()
                for context_id, status in contextids_and_status.items():
                    self._process_contextid(context_id, status)
            except Exception as exc:
                self._logger.error(f"Cleanup exception: {type(exc)}:{exc}")
            finally:
                time.sleep(self._clean_up_interval)

    def _process_contextid(self, context_id: str, status: dict):
        if status["nodes"]:
            if status["released"] or self._is_expired(status["timestamp"]):
                self._cleanup_nodes_list(
                    context_id=context_id, node_ids=status["nodes"]
                )
        else:
            self._remove_contextid_from_cleanup(context_id=context_id)

    def _is_expired(self, timestamp: str):
        now = datetime.now(timezone.utc)
        timestamp = datetime.fromisoformat(timestamp)
        time_elapsed = now - timestamp
        return (
            time_elapsed.seconds > controller_config.cleanup.contextid_release_timelimit
        )

    def _cleanup_nodes_list(self, context_id: str, node_ids: List[str]):
        for node_id in node_ids:
            try:
                node_info = self._get_node_info_by_id(node_id)
                task_handler = _create_node_task_handler(node_info)
                task_handler.clean_up(
                    request_id=CLEANER_REQUEST_ID,
                    context_id=context_id,
                )
                self._remove_nodeid_from_cleanup(context_id=context_id, node_id=node_id)
                self._logger.debug(
                    f"clean_up task succeeded for {node_id=} for {context_id=}"
                )
            except Exception as exc:
                self._logger.error(
                    f"clean_up task FAILED for {node_id=} "
                    f"for {context_id=}. Will retry in {self._clean_up_interval=} secs. Fail "
                    f"reason: {type(exc)}:{exc}"
                )

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
        self._cleanup_file_processor.append_to_cleanup_file(
            context_id=context_id, node_ids=algo_execution_node_ids
        )

    def _remove_contextid_from_cleanup(self, context_id: str):
        self._cleanup_file_processor.remove_from_cleanup_file(context_id=context_id)

    def _remove_nodeid_from_cleanup(self, context_id: str, node_id: str):
        self._cleanup_file_processor.remove_from_cleanup_file(
            context_id=context_id, node_id=node_id
        )

    def release_contextid_for_cleanup(self, context_id: str):
        self._cleanup_file_processor.set_released_true_to_file(context_id=context_id)

    def _get_node_info_by_id(self, node_id: str) -> _NodeInfoDTO:
        global_node = NodeLandscapeAggregator().get_global_node()
        local_nodes = NodeLandscapeAggregator().get_all_local_nodes()

        if node_id == global_node.id:
            return _NodeInfoDTO(
                node_id=global_node.id,
                queue_address=":".join([str(global_node.ip), str(global_node.port)]),
                db_address=":".join([str(global_node.db_ip), str(global_node.db_port)]),
                tasks_timeout=controller_config.rabbitmq.celery_tasks_timeout,
                run_udf_task_timeout=controller_config.rabbitmq.celery_run_udf_task_timeout,
            )

        if node_id in local_nodes.keys():
            local_node = local_nodes[node_id]
            return _NodeInfoDTO(
                node_id=local_node.id,
                queue_address=":".join([str(local_node.ip), str(local_node.port)]),
                db_address=":".join([str(local_node.db_ip), str(local_node.db_port)]),
                tasks_timeout=controller_config.rabbitmq.celery_tasks_timeout,
                run_udf_task_timeout=controller_config.rabbitmq.celery_run_udf_task_timeout,
            )

        raise KeyError(
            f"(Cleaner::_get_node_info_by_id) Node with id '{node_id}' is "
            f"not currently available."
        )

    # This is only supposed to be called from a test.
    # In all other circumstances cleanup should not be reset manually
    def _reset_cleanup(self):
        self._cleanup_file_processor._initialize_cleanup_file()


def _create_node_task_handler(node_info: _NodeInfoDTO) -> NodeAlgorithmTasksHandler:
    return NodeAlgorithmTasksHandler(
        node_id=node_info.node_id,
        node_queue_addr=node_info.queue_address,
        node_db_addr=node_info.db_address,
        tasks_timeout=node_info.tasks_timeout,
        run_udf_task_timeout=node_info.run_udf_task_timeout,
    )


class CleanupFileProcessor:
    def __init__(self, logger):
        self._file_lock = Lock()
        self._logger = logger

        # Create all parent folders, if needed.
        Path(controller_config.cleanup.contextids_cleanup_folder).mkdir(
            parents=True, exist_ok=True
        )

        self._cleanup_file_path = Path(
            controller_config.cleanup.contextids_cleanup_folder
        ).joinpath(Path(CONTEXT_ID_CLEANUP_FILE))

        # create file if it does not exist
        if not os.path.isfile(self._cleanup_file_path):
            self._initialize_cleanup_file()

        # changes to the file will be first written on a temporary file. Then the temporary
        # file replaces the cleanup file. The reason for that is that renaming is atomic
        # (under linux) so the chances of the cleanup file to get corrupted is minimized
        dirname = os.path.dirname(self._cleanup_file_path)
        filename_tmp = (
            Path(self._cleanup_file_path).stem
            + "_tmp"
            + Path(self._cleanup_file_path).suffix
        )
        self._cleanup_file_tmp_path = os.path.join(dirname, filename_tmp)

    def append_to_cleanup_file(self, context_id: str, node_ids: List[str]):
        parsed_toml = self.read_cleanup_file()
        if context_id not in parsed_toml:
            parsed_toml[context_id] = {"nodes": node_ids}
        else:
            self._logger.warning(
                f"(Cleaner::append_to_cleanup_file) Attempting to add {context_id=} for "
                f"cleanup but this context_id is already in the contextids_cleanup_file. "
                f"This should never happen..."
            )
            parsed_toml[context_id]["nodes"].extend(node_ids)
            # remove possible duplicates
            parsed_toml[context_id]["nodes"] = list(
                set(parsed_toml[context_id]["nodes"])
            )

        now_timestamp = datetime.now(timezone.utc)
        parsed_toml[context_id]["timestamp"] = now_timestamp.isoformat()
        parsed_toml[context_id]["released"] = False

        self._write_to_cleanup_file(parsed_toml)

    def set_released_true_to_file(self, context_id: str):
        parsed_toml = self.read_cleanup_file()
        if context_id in parsed_toml:
            parsed_toml[context_id]["released"] = True

        self._write_to_cleanup_file(parsed_toml)

    def remove_from_cleanup_file(self, context_id: str, node_id: str = None):
        parsed_toml = self.read_cleanup_file()
        if context_id in parsed_toml:
            if node_id:
                try:
                    parsed_toml[context_id]["nodes"].remove(node_id)
                except ValueError:
                    self._logger.warning(
                        f"(Cleaner::remove_from_cleanup_file) Tried to remove {node_id=} "
                        f"for {context_id=} but this context_id.node_id is not in the "
                        f"clean_up file.This should not happen."
                    )
                    pass
            else:
                parsed_toml.pop(context_id)
        else:
            self._logger.warning(
                f"(Cleaner::remove_from_cleanup_file) Tried to remove {context_id=} but "
                f"this context_id is not in the clean_up file.This should not happen."
            )
            pass

        self._write_to_cleanup_file(parsed_toml)

    def read_cleanup_file(self) -> dict:
        with self._file_lock:
            if not os.path.isfile(self._cleanup_file_path):
                self._logger.warning(
                    f"(Cleaner::read_cleanup_file) {self._cleanup_file_path=} does not "
                    f"exist. This should not happen"
                )
                return {}
            with open(self._cleanup_file_path, "r") as f:
                try:
                    parsed_toml = toml.load(f)
                except Exception as exc:
                    self._logger.warning(
                        f"(Cleaner::read_cleanup_file) Trying to read {controller_config.cleanup.contextids_cleanup_file=} "
                        f"raised exception: {exc}"
                    )
            return parsed_toml

    def _write_to_cleanup_file(self, toml_string: str):
        with self._file_lock:
            with open(self._cleanup_file_tmp_path, "w") as f:
                toml.dump(toml_string, f)

            # renaming is atomic. Of course if more Controllers are spawn the file is very
            # likely to become corrupted
            os.rename(self._cleanup_file_tmp_path, self._cleanup_file_path)

    def _initialize_cleanup_file(self):
        with self._file_lock:
            # delete it
            cleanup_file_path = Path(
                controller_config.cleanup.contextids_cleanup_folder
            ).joinpath(Path(CONTEXT_ID_CLEANUP_FILE))
            cleanup_file_path.unlink(missing_ok=True)
            # create it
            Path(self._cleanup_file_path).touch()
