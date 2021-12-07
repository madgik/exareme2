import re
import pytest

from mipengine.node.node_logger import initialise_logger
from unittest.mock import patch
from mipengine import AttrDict
from mipengine.node import node_logger


@pytest.fixture(scope="module", autouse=True)
def mock_node_config():
    node_config = AttrDict(
        {
            "identifier": "localnode1",
            "log_level": "INFO",
            "role": "LOCALNODE",
        }
    )

    with patch(
        "mipengine.node.node_logger.node_config",
        node_config,
    ):
        yield

    return node_config


@initialise_logger
def pass_ctx_id(context_id):
    logger = node_logger.get_logger()
    logger.info("Yolo!")
    return logger


def test_get_ctx_id_from_args(capsys):
    test_ctx_id = pass_ctx_id("1234abcd")
    captured = capsys.readouterr()

    # regex to check timestamp
    my_regex = re.compile(r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}\s-[^"]*')
    assert my_regex.match(captured.err) is not None
    assert captured.err.find(" INFO ") > -1
    assert captured.err.find(" NODE ") > -1
    assert captured.err.find("1234abcd") > -1
    assert test_ctx_id.name == "node"
    assert test_ctx_id.level == 20
