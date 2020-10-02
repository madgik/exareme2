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


######### algorithm params:
# algorithm: the algorithm module
# parameters: the algorithm parameters
# attributes: the attributes on which the algorithm will run
# db_objects: the connection objects and the database names
# localtable: the name of the result table in localnodes
# globaltable: the name of the table in global node which contains the merge of all the local result tables
# viewlocaltable: the view which contains the data that will be processed
# globalresulttable: the name of the result table in globalnode
###########
# this function gets the dataflow definition from the [algorithm].py file and replaces local and global calls with the corrresponding calls that are implemented by the system
# which handle the database objects and the parallelism. This is not called in the current implementation but it gives only a description of  the logic.
### Currently it is done with regular expression and this is the bad way.
### Other options are using Python's parser module to edit the byte code but this is monkey patching and requires a lot of attention, or using classes and decorators
## but in this case several issues occur (e.g., should an algorithm developed outside the system be able to have access to the system's objects like the connection objects of the DBs?)
## Another perhaps cleaner option is to support a subset of Python or any other dataflow language which is enough for a developer to use the system defined tasks and produce any possible dataflow.
## Currently there are 2 system defined tasks: 1) _local: runs a task in all the local nodes 2) _global merges local results and runs a task on the global server.
## Other system defined tasks should be added (e.g., run a task to 1 or N local nodes) so that the algorithm developer is able to define any kind of data flow.
## When in production this function probably will act as a parser of a user defined dataflow and an interpreter that interprets this dataflow to the system's internal flow of tasks.
## In this way, we are able to separate the algorithm from the system's internals, so that it is simply an input to the system and agnostic to the techniques  the system uses to implement the dataflows.
async def dataflow_parse_and_execute(task_runner):
    dataflow_source_input = inspect.getsource(task_runner.algorithm.dataflow)
    dataflow_func = [0]

    ###### of course this MUST change ########################
    edited_source = re.sub(
        "self\.\_global\(iternum(?:\s)*\,(?:\s)*globaltable(?:\s)*\,(?:\s)*parameters(?:\s)*\,(?:\s)*attributes(?:\s)*\)",
        "await task_runner._global(iternum)",
        dataflow_source_input,
    )
    edited_source = re.sub(
        "self\.\_local\(iternum(?:\s)*\,(?:\s)*viewlocaltable(?:\s)*\,(?:\s)*parameters(?:\s)*\,(?:\s)*attributes(?:\s)*,(?:\s)*globalresulttable\)",
        "await task_runner._local(iternum)",
        edited_source,
    )
    edited_source = edited_source.split("\n", 1)[-1]
    edited_source = (
        "async def dataflow(task_runner):\n"
        + edited_source
        + "\ndataflow_func[0] = dataflow"
    )
    #########################################################################################

    abstract_syntax_tree = parser.suite(edited_source)
    exec(parser.compilest(abstract_syntax_tree))
    return await dataflow_func[0](task_runner)


# this is an example dataflow for pearson, this function is not called in the current source, to run pearson you should rename this function
# to dataflow and rename the below dataflow function to something different
async def dataflow2(task_runner):
    iternum = 0
    await task_runner._local(iternum)
    return await task_runner._global(iternum)


async def dataflow(task_runner):    #### this  is an example dataflow for countiter
        res = 0
        for iternum in range(100):
            await task_runner._local(iternum)
            res = await task_runner._global(iternum)
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
    task_runner = task.Task(db_objects, unique_id, algorithm_instance, params)
    transfer_runner = transfer.Transfer(db_objects, unique_id)

    # create database views on local databases - each view processes the filters and the selected attributes on the requested table
    # the algorithm won't run directly on the local dataset but on the view
    await task_runner.createlocalviews()

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
                await task_runner.initialize(
                    algorithm_metadata["algorithms"][c]["local_schema"],
                    algorithm_metadata["algorithms"][c]["global_schema"],
                )
                #### run the algorithm dataflow
                result = await dataflow(task_runner)
            except:

                #### clean unused tables
                await task_runner.clean_up()
                raise
    ### clean up tables that are created during the execution
    await task_runner.clean_up()
    return result

