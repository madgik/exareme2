import datetime
import random
import task_executor
import transfer
import json
import importlib
import parser
import symbol, token
import inspect
import re
import settings
import scheduler

DEBUG = settings.DEBUG

def get_package(algorithm):
    try:
        mpackage = "algorithms"
        importlib.import_module(mpackage)
        algo = importlib.import_module("." + algorithm, mpackage)
        if DEBUG:
            importlib.reload(algo)
    except ModuleNotFoundError:
        raise Exception(f"`{algorithm}` does not exist in the algorithms library")
    return algo

def get_uniquetableid():
    return "user{0}".format(
        datetime.datetime.now().microsecond + (random.randrange(1, 100 + 1) * 100000)
    )

async def run(algorithm, params, db_objects):
    result = []
    params = json.loads(params)
    ### get the corresponding algorithm python module using algorithm name
    module = get_package(algorithm)
    algorithm_instance = module.Algorithm()
    table_id = get_uniquetableid()
    transfer_runner = transfer.Transfer(db_objects, table_id)
    task_executor_instance = task_executor.Task(db_objects, table_id, params, transfer_runner)
    # create database views on local databases - each view processes the filters and the selected attributes on the requested table
    # the algorithm won't run directly on the local dataset but on the view
    await task_executor_instance.createlocalviews()
    try:
        scheduler_instance  = scheduler.Scheduler(task_executor_instance, algorithm_instance.algorithm)
        result = await scheduler_instance.schedule()
    except:
        await task_executor_instance.clean_up()
        raise
    await task_executor_instance.clean_up()
    return result