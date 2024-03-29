import re
import subprocess
from unittest.mock import patch

import pytest

from exareme2 import AttrDict
from exareme2.controller.federation_info_logs import log_experiment_execution
from exareme2.controller.logger import init_logger
from exareme2.controller.services.node_landscape_aggregator import (
    _log_data_model_changes,
)
from exareme2.controller.services.node_landscape_aggregator import _log_dataset_changes
from exareme2.controller.services.node_landscape_aggregator import _log_node_changes
from exareme2.node_communication import NodeInfo
from tests.standalone_tests.conftest import MONETDB_LOCALNODETMP_PORT
from tests.standalone_tests.conftest import MonetDBConfigurations

LOGFILE_NAME = "test_show_controller_audit_entries.out"


@pytest.mark.slow
@pytest.mark.very_slow
def test_show_node_db_actions(monetdb_localnodetmp, load_data_localnodetmp):
    """
    Load data into the db and then remove datamodel and datasets.
    Assert that the logs produced with federation_info.py contain these changes.
    """
    monet_db_confs = MonetDBConfigurations(port=MONETDB_LOCALNODETMP_PORT)
    cmd = f'mipdb delete-data-model dementia -v "0.1" {monet_db_confs.convert_to_mipdb_format()} --force'
    res = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert res.returncode == 0

    cmd = f"python3 federation_info.py show-node-db-actions --port {MONETDB_LOCALNODETMP_PORT}"
    res = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    output = str(res.stdout)
    assert re.match(r".*\\n.* - .* - ADD DATA MODEL - .* - .*\\n.*", output)
    assert re.match(r".*\\n.* - .* - DELETE DATA MODEL - .* - .*\\n.*", output)
    assert re.match(r".*\\n.* - .* - ADD DATASET - .* - .* - .* - .*\\n.*", output)
    assert re.match(r".*\\n.* - .* - DELETE DATASET - .* - .* - .* - .*\\n.*", output)


@pytest.fixture(scope="session")
def controller_config_dict_mock():
    controller_config = {
        "log_level": "INFO",
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
    _log_node_changes(
        old_nodes=[
            NodeInfo(
                id="localnode1",
                role="LOCALNODE",
                ip="172.17.0.1",
                port=60001,
                db_ip="172.17.0.1",
                db_port=61001,
            )
        ],
        new_nodes=[
            NodeInfo(
                id="localnode2",
                role="LOCALNODE",
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
        old_datasets_locations={"dementia:0.1": {"edsd": "localnode1"}},
        new_datasets_locations={"tbi:0.1": {"dummy_tbi": "localnode2"}},
        logger=logger,
    )
    log_experiment_execution(
        logger=logger,
        request_id="test",
        context_id="test_cntxtid",
        algorithm_name="test_algorithm",
        datasets=["edsd"],
        algorithm_parameters="parameters",
        local_node_ids=["localnode1"],
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
    assert re.match(r".* - NODE_JOINED - localnode2\\n.*", output)
    assert re.match(r".*\\n.* - NODE_LEFT - localnode1\\n.*", output)
    assert re.match(r".*\\n.* - DATAMODEL_ADDED - tbi:0.1\\n.*", output)
    assert re.match(r".*\\n.* - DATAMODEL_REMOVED - dementia:0.1\\n.*", output)
    assert re.match(
        r".*\\n.* - DATASET_ADDED - localnode2 - tbi:0.1 - dummy_tbi\\n.*", output
    )
    assert re.match(
        r".*\\n.* - DATASET_REMOVED - localnode1 - dementia:0.1 - edsd\\n.*", output
    )
    assert re.match(
        r".*\\n.* - EXPERIMENT_STARTED - test - test_algorithm - edsd - parameters\\n.*",
        output,
    )
