import re
from unittest.mock import patch

import pytest

from mipengine import AttrDict
from mipengine.node import logging
from mipengine.node.logging import log_method_call


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
        "mipengine.node.logging.node_config",
        node_config,
    ):
        yield

    return node_config


def test_log_format(capsys):
    logger = logging.getLogger(__name__)
    logger.info("this is a test")

    captured = capsys.readouterr()
    # regex to check timestamp
    my_regex = re.compile(r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}\s-[^"]*')
    assert my_regex.match(captured.out) is not None
    assert captured.out.find(" INFO ") > -1
    assert captured.out.find(" localnode1 ") > -1
    assert captured.out.find(" test_log_format") > -1
    assert captured.out.find(" this is a test") > -1
    assert captured.out.find(" NODE ") > -1
    assert captured.out.find(" LOCALNODE ") > -1


def test_decorator(capsys):
    @log_method_call
    def log_method_test():
        pass

    log_method_test()
    captured = capsys.readouterr()
    assert captured.out.find("log_method_test method started") != -1
    assert captured.out.find("log_method_test method succeeded") != -1
