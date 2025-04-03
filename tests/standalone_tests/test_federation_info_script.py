import re
import subprocess
from unittest.mock import patch

import pytest

from exareme2 import AttrDict
from exareme2.controller.federation_info_logs import log_experiment_execution
from exareme2.controller.logger import init_logger
from exareme2.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    _log_data_model_changes,
)
from exareme2.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    _log_dataset_changes,
)
from exareme2.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    _log_worker_changes,
)
from exareme2.worker_communication import WorkerInfo
from tests.standalone_tests.conftest import MONETDB_LOCALWORKERTMP_PORT
from tests.standalone_tests.conftest import TEST_DATA_FOLDER
from tests.standalone_tests.conftest import MonetDBConfigurations

LOGFILE_NAME = "test_show_controller_audit_entries.out"


@pytest.fixture(scope="session")
def controller_config_dict_mock():
    controller_config = {
        "log_level": "INFO",
        "node_identifier": "controller",
        "federation": "standalone_tests",
    }
    yield controller_config


@pytest.fixture(scope="session")
def patch_controller_logger_config(controller_config_dict_mock):
    with patch(
        "exareme2.controller.logger.ctrl_config",
        AttrDict(controller_config_dict_mock),
    ):
        yield


@pytest.mark.slow
def test_show_controller_audit_entries(patch_controller_logger_config, capsys):
    logger = init_logger("BACKGROUND")
    _log_worker_changes(
        old_workers=[
            WorkerInfo(
                id="localworker1",
                role="LOCALWORKER",
                ip="172.17.0.1",
                port=60001,
                db_ip="172.17.0.1",
                db_port=61001,
            )
        ],
        new_workers=[
            WorkerInfo(
                id="localworker2",
                role="LOCALWORKER",
                ip="172.17.0.1",
                port=60002,
                db_ip="172.17.0.1",
                db_port=61002,
            )
        ],
        logger=logger,
    )
    _log_data_model_changes(
        old_data_models={"dementia:0.1": ""},
        new_data_models={"tbi:0.1": ""},
        logger=logger,
    )
    _log_dataset_changes(
        old_datasets_locations={"dementia:0.1": {"edsd": "localworker1"}},
        new_datasets_locations={"tbi:0.1": {"dummy_tbi": "localworker2"}},
        logger=logger,
    )
    log_experiment_execution(
        logger=logger,
        request_id="test",
        context_id="test_cntxtid",
        algorithm_name="test_algorithm",
        datasets=["edsd"],
        algorithm_parameters="parameters",
        local_worker_ids=["localworker1"],
    )

    # Get the logged output
    captured = capsys.readouterr()

    cmd = f"python3 federation_info.py show-controller-audit-entries"
    res = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        input=captured.err.encode(),
    )

    output = str(res.stdout)
    assert re.match(r".* - WORKER_JOINED - localworker2\\n.*", output)
    assert re.match(r".*\\n.* - WORKER_LEFT - localworker1\\n.*", output)
    assert re.match(r".*\\n.* - DATAMODEL_ADDED - tbi:0.1\\n.*", output)
    assert re.match(r".*\\n.* - DATAMODEL_REMOVED - dementia:0.1\\n.*", output)
    assert re.match(
        r".*\\n.* - DATASET_ADDED - localworker2 - tbi:0.1 - dummy_tbi\\n.*", output
    )
    assert re.match(
        r".*\\n.* - DATASET_REMOVED - localworker1 - dementia:0.1 - edsd\\n.*", output
    )
    assert re.match(
        r".*\\n.* - EXPERIMENT_STARTED - test - test_algorithm - edsd - parameters\\n.*",
        output,
    )
