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
import settings

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



#### TODO this function is becoming complicated needs refactoring
async def dataflow_scheduler(task_executor, algorithm):
    localtable = task_executor.localtable
    globaltable = task_executor.globaltable
    viewlocaltable = task_executor.viewlocaltable
    globalresulttable = task_executor.globalresulttable
    attributes = task_executor.attributes

    ## bind parameters before pushing them to the algorithm - necessary step to avoid sql injections
    parameters = task_executor.parameters
    parameters = task_executor.bindparameters(parameters)
    query_generator = algorithm(viewlocaltable, globaltable, parameters, attributes, globalresulttable)
    local_first_step = 1

    tasks = next(query_generator)
    static_schema = False
    local_schema = None
    global_schema = None

    if tasks[2] == "schema":
        static_schema = True
        local_schema = tasks[0]
        global_schema = tasks[1]
        await task_executor.init_tables(local_schema,global_schema)
        tasks = next(query_generator)
        if tasks[1] == 'local':
            await task_executor.task_local(local_schema, tasks[0])
        elif tasks[1] == 'global':
            local_first_step = 0
            result = await task_executor.task_global(global_schema, tasks[0])
    else:
        sqlscript = tasks[1]
        if tasks[2] == 'local':
            local_schema = tasks[0]
            await task_executor.task_local(local_schema, sqlscript)
        elif tasks[2] == 'global':
            local_first_step = 0
            global_schema = tasks[0]
            result = await task_executor.task_global(global_schema, sqlscript)

    while True:
        try:
            if local_first_step:
                tasks = next(query_generator)
                #####  the following runs only when termination condition is evaluated in SQL
                sqlscript = ''
                schema = global_schema
                if static_schema == True:
                    sqlscript = tasks[0]
                else:
                    sqlscript = tasks[1]
                    schema = tasks[0]

                if 'termination' in schema:
                    ##  TODO: this is quick and dirty - rewrite efficiently
                    result = await task_executor.task_global(schema, sqlscript)
                    if 'iternum' in schema:
                        if result[len(result)-1][0] == True:
                            return [x[2:] for x in result if x[1] == result[len(result)-1][1]]
                    else:
                        if result[0][0] == True:
                            return [x[1:] for x in result]
                    tasks = next(query_generator)
                    sqlscript = ''
                    schema = local_schema
                    if static_schema == True:
                        sqlscript = tasks[0]
                    else:
                        sqlscript = tasks[1]
                        schema = tasks[0]
                    await task_executor.task_local(schema, sqlscript)

                ######################################################################
                else:
                    result = await task_executor.task_global(schema, sqlscript)
                    next(query_generator)
                    tasks = query_generator.send(result)
                    sqlscript = ''
                    schema = local_schema
                    if static_schema == True:
                        sqlscript = tasks[0]
                    else:
                        sqlscript = tasks[1]
                        schema = tasks[0]
                    await task_executor.task_local(schema, sqlscript)
            else:
                if 'termination' in global_schema:
                    ##  TODO: this is quick and dirty - rewrite efficiently
                    tasks= next(query_generator)
                    sqlscript = ''
                    schema = local_schema
                    if static_schema == True:
                        sqlscript = tasks[0]
                    else:
                        sqlscript = tasks[1]
                        schema = tasks[0]
                    await task_executor.task_local(schema, sqlscript)
                    tasks = next(query_generator)
                    sqlscript = ''
                    schema = global_schema
                    if static_schema == True:
                        sqlscript = tasks[0]
                    else:
                        sqlscript = tasks[1]
                        schema = tasks[0]
                    result = await task_executor.task_global(schema, sqlscript)
                    if 'iternum' in schema or 'history' in schema:
                        if result[len(result)-1][0] == True:
                            return [x[2:] for x in result if x[1] == result[len(result)-1][1]]
                    else:
                        if result[0][0] == True:
                            return [x[1:] for x in result]
                else:
                    next(query_generator)
                    tasks = query_generator.send(result)
                    sqlscript = ''
                    schema = local_schema
                    if static_schema == True:
                        sqlscript = tasks[0]
                    else:
                        sqlscript = tasks[1]
                        schema = tasks[0]
                    await task_executor.task_local(schema, sqlscript)
                    tasks = next(query_generator)
                    sqlscript = ''
                    schema = global_schema
                    if static_schema == True:
                        sqlscript = tasks[0]
                    else:
                        sqlscript = tasks[1]
                        schema = tasks[0]
                    result = await task_executor.task_global(schema, sqlscript)
        except StopIteration:
            break
    if static_schema:
        if 'termination' in global_schema and ('iternum' not in global_schema and 'history' not in global_schema):
            return [x[1:] for x in result]
        if 'termination' in global_schema and ('iternum' in global_schema or 'history' in global_schema):
            return [x[2:] for x in result]
        else:
            return result
    else:  ## schema redefined in each step, to support fully
        if 'termination' in schema:
            return [x[1:] for x in result]
        else:
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
        result  = await dataflow_scheduler(task_executor, algorithm_instance.algorithm)
    except:
        await task_executor.clean_up()
        raise
    await task_executor.clean_up()
    return result