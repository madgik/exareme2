import json
import random

from tests.dev_env_tests.nodes_communication import get_celery_app
from tests.dev_env_tests.nodes_communication import get_celery_task_signature

local_node = get_celery_app("localnode1")


def routine(id):
    id = random.randint(0, 123456)
    request_id = "7222076" + id.__str__()
    context_id = "2522180" + id.__str__()
    table_name = f"normal_localnode1_{context_id}_7_0"
    opener = open("input0.txt", "r")
    x = json.loads(opener.read())
    opener.close()

    for task, kwargs in x:
        if not kwargs:
            continue

        if task == "run_udf":
            keyword_args_json = kwargs["keyword_args_json"]
            keyword_args_json = keyword_args_json.replace("2522180", context_id)
            get_celery_task_signature(local_node, task).delay(
                command_id=kwargs["command_id"],
                request_id=request_id,
                context_id=context_id,
                func_name=kwargs["func_name"],
                positional_args_json=kwargs["positional_args_json"],
                keyword_args_json=keyword_args_json,
            ).get()
        elif task == "create_data_model_view":
            columns = kwargs["columns"]
            while "row_id" in columns:
                columns.remove("row_id")
            get_celery_task_signature(local_node, task).delay(
                request_id=request_id,
                context_id=context_id,
                command_id=kwargs["command_id"],
                data_model=kwargs["data_model"],
                datasets=kwargs["datasets"],
                columns=columns,
            ).get()
        elif task == "get_table_data":
            get_celery_task_signature(local_node, task).delay(
                request_id=request_id, table_name=table_name
            ).get()
        elif task == "clean_up":
            get_celery_task_signature(local_node, task).delay(
                request_id=request_id, context_id=context_id
            ).get()
        elif task == "get_node_datasets_per_data_model":
            get_celery_task_signature(local_node, task).delay(
                request_id=request_id
            ).get()
        elif task == "get_node_info":
            get_celery_task_signature(local_node, task).delay(
                request_id=request_id
            ).get()
        elif task == "get_data_model_cdes":
            get_celery_task_signature(local_node, task).delay(
                request_id=request_id, data_model=kwargs["data_model"]
            ).get()


def test_0_0():
    routine(0)


def test_0_1():
    routine(1)


def test_0_2():
    routine(2)


def test_0_3():
    routine(3)


def test_0_4():
    routine(4)


def test_0_5():
    routine(5)


def test_0_6():
    routine(6)


def test_0_7():
    routine(7)


def test_0_8():
    routine(8)


def test_0_9():
    routine(9)


def test_1_0():
    routine(10)


def test_1_1():
    routine(11)


def test_1_2():
    routine(12)


def test_1_3():
    routine(13)


def test_1_4():
    routine(14)


def test_1_5():
    routine(15)


def test_1_6():
    routine(16)


def test_1_7():
    routine(17)


def test_1_8():
    routine(18)


def test_1_9():
    routine(19)


def test_2_0():
    routine(20)


def test_2_1():
    routine(21)


def test_2_2():
    routine(22)


def test_2_3():
    routine(23)


def test_2_4():
    routine(24)


def test_2_5():
    routine(25)


def test_2_6():
    routine(26)


def test_2_7():
    routine(27)


def test_2_8():
    routine(28)


def test_2_9():
    routine(29)


def test_3_0():
    routine(30)


def test_3_1():
    routine(31)


def test_3_2():
    routine(32)


def test_3_3():
    routine(33)


def test_3_4():
    routine(34)


def test_3_5():
    routine(35)


def test_3_6():
    routine(36)


def test_3_7():
    routine(37)


def test_3_8():
    routine(38)


def test_3_9():
    routine(39)


def test_4_0():
    routine(40)


def test_4_1():
    routine(41)


def test_4_2():
    routine(42)


def test_4_3():
    routine(43)


def test_4_4():
    routine(44)


def test_4_5():
    routine(45)


def test_4_6():
    routine(46)


def test_4_7():
    routine(47)


def test_4_8():
    routine(48)


def test_4_9():
    routine(49)


def test_5_0():
    routine(50)


def test_5_1():
    routine(51)


def test_5_2():
    routine(52)


def test_5_3():
    routine(53)


def test_5_4():
    routine(54)


def test_5_5():
    routine(55)


def test_5_6():
    routine(56)


def test_5_7():
    routine(57)


def test_5_8():
    routine(58)


def test_5_9():
    routine(59)


def test_6_0():
    routine(60)


def test_6_1():
    routine(61)


def test_6_2():
    routine(62)


def test_6_3():
    routine(63)


def test_6_4():
    routine(64)


def test_6_5():
    routine(65)


def test_6_6():
    routine(66)


def test_6_7():
    routine(67)


def test_6_8():
    routine(68)


def test_6_9():
    routine(69)


def test_7_0():
    routine(70)


def test_7_1():
    routine(71)


def test_7_2():
    routine(72)


def test_7_3():
    routine(73)


def test_7_4():
    routine(74)


def test_7_5():
    routine(75)


def test_7_6():
    routine(76)


def test_7_7():
    routine(77)


def test_7_8():
    routine(78)


def test_7_9():
    routine(79)


def test_8_0():
    routine(80)


def test_8_1():
    routine(81)


def test_8_2():
    routine(82)


def test_8_3():
    routine(83)


def test_8_4():
    routine(84)


def test_8_5():
    routine(85)


def test_8_6():
    routine(86)


def test_8_7():
    routine(887)


def test_8_8():
    routine(88)


def test_8_9():
    routine(89)


def test_9_0():
    routine(90)


def test_9_1():
    routine(91)


def test_9_2():
    routine(92)


def test_9_3():
    routine(93)


def test_9_4():
    routine(94)


def test_9_5():
    routine(95)


def test_9_6():
    routine(96)


def test_9_7():
    routine(97)


def test_9_8():
    routine(98)


def test_9_9():
    routine(99)
