import re

from mipengine.controller import controller_logger as ctrl_logger
from mipengine.controller.controller_logger import log_method_call


def test_log_format(capsys):
    logger = ctrl_logger.getLogger()
    logger.info("this is a test")

    captured = capsys.readouterr()
    # regex to check timestamp
    my_regex = re.compile(r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}\s-[^"]*')
    assert my_regex.match(captured.out) is not None
    assert captured.out.find(" INFO ") > -1
    assert captured.out.find(" CONTROLLER ") > -1
    assert captured.out.find(" test_log_format") > -1
    assert captured.out.find(" this is a test") > -1


def test_decorator(capsys):
    @log_method_call
    def log_method_test():
        pass

    log_method_test()
    captured = capsys.readouterr()
    assert captured.out.find("log_method_test method started") != -1
    assert captured.out.find("log_method_test method succeeded") != -1
