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



###########
# this function gets the dataflow definition from the [algorithm].py file and replaces local and global calls with the corrresponding calls that 
# run by the task executor and handle the database objects and the parallelism. This is not called in the current implementation but it gives just
# a description of  the logic. Currently it is done with regular expression and this is the bad way.
# Other options are using Python's parser module to edit the byte code but this is monkey patching and requires a lot of attention, or using classes and decorators
# but in this case several issues occur (e.g., should an algorithm developed outside the system be able to have access to the system's objects like the connection objects of the DBs?)
# Another perhaps cleaner option is to support a subset of Python or any other dataflow language which is enough for a developer to use the system defined tasks and produce any possible dataflow.
# Currently there are 2 system defined tasks: 1) _local: runs a task in all the local nodes 2) _global merges local results and runs a task on the global server.
# Other system defined tasks should be added (e.g., run a task to 1 or N local nodes) so that the algorithm developer is able to define any kind of data flow.
# When in production this function probably will act as a parser of a user defined dataflow and an interpreter that interprets this dataflow to the system's internal flow of tasks.
# In this way, we are able to separate the algorithm from the system's internals, so that it is simply an input to the system and agnostic to the techniques  the system uses to implement the dataflows.
async def dataflow_parse_and_execute(dataflow,  task_executor):
    dataflow_source_input = inspect.getsource(dataflow)
    dataflow_func = [0]

    ###### of course this MUST change ########################
    edited_source = re.sub(
        "self\.\_global\(iternum(?:\s)*\,(?:\s)*globaltable(?:\s)*\,(?:\s)*parameters(?:\s)*\,(?:\s)*attributes(?:\s)*\)",
        "await task_executor._global(iternum)",
        dataflow_source_input,
    )
    edited_source = re.sub(
        "self\.\_local\(iternum(?:\s)*\,(?:\s)*viewlocaltable(?:\s)*\,(?:\s)*parameters(?:\s)*\,(?:\s)*attributes(?:\s)*,(?:\s)*globalresulttable\)",
        "await task_executor._local(iternum)",
        edited_source,
    )
    edited_source = edited_source.split("\n", 1)[-1]
    edited_source = (
        "async def dataflow(task_executor):\n"
        + edited_source
        + "\ndataflow_func[0] = dataflow"
    )
    #########################################################################################

    abstract_syntax_tree = parser.suite(edited_source)
    exec(parser.compilest(abstract_syntax_tree))
    return await dataflow_func[0](task_executor)


# this is an example dataflow for pearson, this function is not called in the current execution flow, 
# to run pearson you should edit the call to dataflow function in run
async def dataflow_pearson(task_executor):
    iternum = 0
    await task_executor._local(iternum)
    return await task_executor._global(iternum)


async def dataflow_countiter(task_executor):    #### this  is an example dataflow for countiter
        res = 0
        for iternum in range(100):
            await task_executor._local(iternum)
            res = await task_executor._global(iternum)
            if res[0][0] > 1000000:
                break
        return res

#### run function:
# creates unique table names
# get the algorithm name and accesses the corresponding python module
# creates the local views
# gets also db_objects: the connection objects to all the nodes of the federation
# initializes the connections between the servers and the remote/merge tables
# runs the dataflow
# cleans up the servers and returns the results


async def run(algorithm, params, db_objects):
    result = []
    params = json.loads(params)

    ### get the corresponding algorithm python module using algorithm name

    module = get_package(algorithm)
    algorithm_instance = module.Algorithm()
    unique_id = get_uniquetableid()
    task_executor = task.Task(db_objects, unique_id, algorithm_instance, params)
    transfer_runner = transfer.Transfer(db_objects, unique_id)

    # create database views on local databases - each view processes the filters and the selected attributes on the requested table
    # the algorithm won't run directly on the local dataset but on the view
    await task_executor.createlocalviews()

    ##### schema.json contains info about each algorithm: the name and the intermediate result schema
    with open("schema.json") as properties:
        algorithm_metadata = json.load(properties)

    for c, algo in enumerate(
        [
            algorithm_metadata["algorithms"][i]["name"]
            for i, j in enumerate(algorithm_metadata["algorithms"])
        ]
    ):
        if algorithm == algo:
            try:
                ### initialize the connections between the databases, this runs only with postgres at the time, in case of monetdb it has no impact
                await transfer_runner.initialize(
                    algorithm_metadata["algorithms"][c]["local_schema"],
                    algorithm_metadata["algorithms"][c]["global_schema"],
                )
                #### initialize database tables
                await task_executor.initialize(
                    algorithm_metadata["algorithms"][c]["local_schema"],
                    algorithm_metadata["algorithms"][c]["global_schema"],
                )
                #### run the algorithm dataflow
                result = await dataflow_countiter(task_executor)
                #result = await dataflow_pearson(task_executor)
                #result =  await dataflow_parse_and_execute(algorithm_instance.dataflow, task_executor)
            except:

                #### clean unused tables
                await task_executor.clean_up()
                raise
    ### clean up tables that are created during the execution
    await task_executor.clean_up()
    return result

