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


async def dataflow(task_executor, algorithm):

    localtable = task_executor.localtable
    globaltable = task_executor.globaltable
    viewlocaltable = task_executor.viewlocaltable
    globalresulttable = task_executor.globalresulttable
    attributes = task_executor.attributes

    ## bind parameters before pushing them to the algorithm - necessary step to avoid sql injections
    parameters = task_executor.parameters
    parameters = task_executor.bindparameters(parameters)

    query_generator = algorithm(viewlocaltable, globaltable, parameters, attributes, globalresulttable)

    schema, sqlscript = next(query_generator)
    await task_executor._local(schema, sqlscript)
    while True:
        try:
            schema, sqlscript = next(query_generator)
            result = await task_executor._global(schema, sqlscript)
            next(query_generator)
            schema, sqlscript = query_generator.send(result)
            await task_executor._local(schema, sqlscript)
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
    task_executor = task.Task(db_objects, table_id, algorithm_instance, params, transfer_runner)

    # create database views on local databases - each view processes the filters and the selected attributes on the requested table
    # the algorithm won't run directly on the local dataset but on the view
    await task_executor.createlocalviews()

    try:
        result  = await dataflow(task_executor, algorithm_instance.algorithm)
    except:
        #### clean unused tables
        await task_executor.clean_up()
        raise
    ### clean up tables that are created during the execution
    await task_executor.clean_up()
    return result

