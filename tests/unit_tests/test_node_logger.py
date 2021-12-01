from mipengine.node import node_logger
from mipengine.node.node_logger import log_add_ctx_id
from mipengine.node import config as node_config
from unittest.mock import patch

import pytest

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


def test_get_ctx_id_from_args():
    @log_add_ctx_id
    def pass_ctx_id(ctx_id):
        logger = node_logger.get_logger()
        logger.info("Helloooooooooooooooooo")

    pass_ctx_id("123456node")
