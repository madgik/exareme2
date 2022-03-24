import asyncio
import os
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import List

import toml
from pydantic import BaseModel

from mipengine.controller import config as controller_config
from mipengine.controller import controller_logger as ctrl_logger
from mipengine.controller.node_registry import NodeRegistry
from mipengine.controller.node_tasks_handler_celery import NodeTasksHandlerCelery

CLEANER_REQUEST_ID = "CLEANER"


class _NodeInfoDTO(BaseModel):
    node_id: str
    queue_address: str
    db_address: str
    tasks_timeout: int

    class Config:
        allow_mutation = False


class Cleaner:
    def __init__(self, node_registry: NodeRegistry):
        self._logger = ctrl_logger.get_background_service_logger()

        self._node_registry = node_registry

        self._cleanup_file_processor = CleanupFileProcessor(self._logger)
        self._clean_up_interval = controller_config.cleanup.nodes_cleanup_interval
        self.keep_cleaning_up = True

    async def cleanup_loop(self):
        while self.keep_cleaning_up:
            contextids_and_status = self._cleanup_file_processor._read_cleanup_file()
            for context_id, status in contextids_and_status.items():
                if not status["nodes"]:
                    self._remove_contextid_from_cleanup(context_id=context_id)
                    continue
                if (
                    status["released"]
                    or (datetime.now(timezone.utc) - status["timestamp"]).seconds
                    > controller_config.cleanup.contextid_release_timelimit
                ):
                    for node_id in status["nodes"]:
                        try:
                            node_info = self._get_node_info_by_id(node_id)
                            task_handler = _create_node_task_handler(node_info)
                            task_handler.clean_up(
                                request_id=CLEANER_REQUEST_ID,
                                context_id=context_id,
                            )
                            self._remove_nodeid_from_cleanup(
                                context_id=context_id, node_id=node_id
                            )
                            self._logger.debug(
                                f"clean_up task succeeded for {node_id=} for {context_id=}"
                            )
                        except Exception as exc:
                            self._logger.debug(
                                f"clean_up task FAILED for {node_id=} "
                                f"for {context_id=}. Will retry in a while... fail "
                                f"reason: {type(exc)}:{exc}"
                            )

            await asyncio.sleep(self._clean_up_interval)

    def _add_contextid_for_cleanup(
        self, context_id: str, algo_execution_node_ids: List[str]
    ):
        self._cleanup_file_processor._append_to_cleanup_file(
            context_id=context_id, node_ids=algo_execution_node_ids
        )

    def _remove_contextid_from_cleanup(self, context_id: str):
        self._cleanup_file_processor._remove_from_cleanup_file(context_id=context_id)

    def _remove_nodeid_from_cleanup(self, context_id: str, node_id: str):
        self._cleanup_file_processor._remove_from_cleanup_file(
            context_id=context_id, node_id=node_id
        )

    def _release_contextid_for_cleanup(self, context_id: str):
        self._cleanup_file_processor._set_released_true_to_file(context_id=context_id)

    def _get_node_info_by_id(self, node_id: str) -> _NodeInfoDTO:
        global_nodes = self._node_registry.get_all_global_nodes()
        local_nodes = self._node_registry.get_all_local_nodes()

        for node in global_nodes + local_nodes:
            if node.id == node_id:
                return _NodeInfoDTO(
                    node_id=node.id,
                    queue_address=":".join([str(node.ip), str(node.port)]),
                    db_address=":".join([str(node.db_ip), str(node.db_port)]),
                    tasks_timeout=controller_config.rabbitmq.celery_tasks_timeout,
                )


def _create_node_task_handler(node_info: _NodeInfoDTO) -> NodeTasksHandlerCelery:
    return NodeTasksHandlerCelery(
        node_id=node_info.node_id,
        node_queue_addr=node_info.queue_address,
        node_db_addr=node_info.db_address,
        tasks_timeout=node_info.tasks_timeout,
    )


class CleanupFileProcessor:
    def __init__(self, logger):
        self._logger = logger
        if not os.path.isfile(controller_config.cleanup.contextids_cleanup_file):
            Path(controller_config.cleanup.contextids_cleanup_file).touch()

    def _append_to_cleanup_file(self, context_id: str, node_ids: List[str]):
        dirname = os.path.dirname(controller_config.cleanup.contextids_cleanup_file)
        filename = os.path.basename(controller_config.cleanup.contextids_cleanup_file)
        filename_tmp = Path(filename).stem + "_tmp" + Path(filename).suffix
        with open(controller_config.cleanup.contextids_cleanup_file, "r") as f:
            parsed_toml = toml.load(f)
        if context_id not in parsed_toml:
            parsed_toml[context_id] = {"nodes": node_ids}
        else:
            self._logger.warning(
                f"Attempting to add {context_id=} for cleanup but this context_id is "
                f"already in the contextids_cleanup_file. This should never happen..."
            )
            parsed_toml[context_id]["nodes"].extend(node_ids)
            # remove possible duplicates
            parsed_toml[context_id]["nodes"] = list(
                set(parsed_toml[context_id]["nodes"])
            )

        now_timestamp = datetime.now(timezone.utc)
        parsed_toml[context_id]["timestamp"] = now_timestamp.isoformat()
        parsed_toml[context_id]["released"] = False

        tmp_contextids_cleanup_file = os.path.join(dirname, filename_tmp)
        with open(tmp_contextids_cleanup_file, "w") as f:
            toml.dump(parsed_toml, f)

        # renaming is atomic. Of course if more Controllers are spawn the file is very
        # likely to become corrupted
        os.rename(
            tmp_contextids_cleanup_file,
            controller_config.cleanup.contextids_cleanup_file,
        )

    def _set_released_true_to_file(self, context_id: str):
        dirname = os.path.dirname(controller_config.cleanup.contextids_cleanup_file)
        filename = os.path.basename(controller_config.cleanup.contextids_cleanup_file)
        filename_tmp = "contextids_cleanup_tmp.toml"
        with open(controller_config.cleanup.contextids_cleanup_file, "r") as f:
            parsed_toml = toml.load(f)
        if context_id in parsed_toml:
            parsed_toml[context_id]["released"] = True
        tmp_contextids_cleanup_file = os.path.join(dirname, filename_tmp)
        with open(tmp_contextids_cleanup_file, "w") as f:
            toml.dump(parsed_toml, f)

        # renaming is atomic. Of course if more Controllers are spawn the file is very
        # likely to become corrupted
        os.rename(
            tmp_contextids_cleanup_file,
            controller_config.cleanup.contextids_cleanup_file,
        )

    def _remove_from_cleanup_file(self, context_id: str, node_id: str = None):
        dirname = os.path.dirname(controller_config.cleanup.contextids_cleanup_file)
        filename = os.path.basename(controller_config.cleanup.contextids_cleanup_file)
        filename_tmp = "contextids_cleanup_tmp.toml"
        with open(controller_config.cleanup.contextids_cleanup_file, "r") as f:
            parsed_toml = toml.load(f)
        if context_id in parsed_toml:
            if node_id:
                try:
                    parsed_toml[context_id]["nodes"].remove(node_id)
                except ValueError:
                    # warning if nodeid is not there
                    pass
            else:
                parsed_toml.pop(context_id)
        else:
            # warning if contextid not there
            pass

        tmp_contextids_cleanup_file = os.path.join(dirname, filename_tmp)
        with open(tmp_contextids_cleanup_file, "w") as f:
            toml.dump(parsed_toml, f)

        # renaming is atomic. Of course if more Controllers are spawn the file is very
        # likely to become corrupted
        os.rename(
            tmp_contextids_cleanup_file,
            controller_config.cleanup.contextids_cleanup_file,
        )

    def _read_cleanup_file(self) -> dict:
        with open(controller_config.cleanup.contextids_cleanup_file, "r") as f:
            parsed_toml = toml.load(f)
            for context_id in parsed_toml.keys():
                parsed_toml[context_id]["timestamp"] = datetime.fromisoformat(
                    parsed_toml[context_id]["timestamp"]
                )
            return parsed_toml
