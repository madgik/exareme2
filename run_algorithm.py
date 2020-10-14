import datetime
import random
import task
import transfer
import json
import importlib
import parser
import symbol, token
import inspect
import re


def get_package(algorithm):
    mpackage = "algorithms"
    importlib.import_module(mpackage)
    algo = importlib.import_module("." + algorithm, mpackage)
    return algo


def get_uniquetableid():
    return "user{0}".format(
        datetime.datetime.now().microsecond + (random.randrange(1, 100 + 1) * 100000)
    )





async def dataflow(task_executor, algorithm, params, table_id):
    localtable = "local" + table_id
    globaltable = "global" + table_id
    viewlocaltable = "localview" + table_id
    globalresulttable = "globalres" + table_id
    attributes = params['attributes']
    parameters = params['parameters']

    queries = algorithm(viewlocaltable, globaltable, parameters, attributes, globalresulttable)
    await task_executor._local(*next(queries))
    result = await task_executor._global(*next(queries))
    while True:
        try:
            queries.send(None)
            await task_executor._local(*queries.send(result))
            result = await task_executor._global(*next(queries))
        except StopIteration:
            break

    return result


async def run(algorithm, params, db_objects):
    result = []
    params = json.loads(params)

    ### get the corresponding algorithm python module using algorithm name

    module = get_package(algorithm)
    algorithm_instance = module.Algorithm()
    table_id = get_uniquetableid()
    transfer_runner = transfer.Transfer(db_objects, table_id)
    task_executor = task.Task(db_objects, table_id, algorithm_instance, params,  transfer_runner)

    # create database views on local databases - each view processes the filters and the selected attributes on the requested table
    # the algorithm won't run directly on the local dataset but on the view
    await task_executor.createlocalviews()

    try:
        result  = await dataflow(task_executor, algorithm_instance.algorithm, params, table_id)
    except:
        #### clean unused tables
        await task_executor.clean_up()
        raise
    ### clean up tables that are created during the execution
    await task_executor.clean_up()
    return result

