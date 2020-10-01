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


def get_uniquetablename():
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
# which handle the database objects and the parallelism
### this must be done via editing the bytecode for example using the parser module. Currently it is done with regular expression and this is the bad but fast way.
### Apparently, we could (and in my opinion should) support dataflow definitions in other workflow languages which are designed to support dataflow definitions.


async def dataflow(
    algorithm,
    parameters,
    attributes,
    db_objects,
    localtable,
    globaltable,
    viewlocaltable,
    globalresulttable=None,
):
    src = inspect.getsource(algorithm.dataflow)
    func = [0]

    ###### of course this MUST change and use Python's parser module to edit bytecode ########################
    src = re.sub(
        "\_global\(iternum(?:\s)*\,(?:\s)*globaltable(?:\s)*\,(?:\s)*parameters(?:\s)*\,(?:\s)*attributes(?:\s)*\)",
        "await task._global(iternum, globaltable, parameters, attributes, db_objects, localtable, globalresulttable, algorithm, viewlocaltable)",
        src,
    )
    src = re.sub(
        "\_local\(iternum(?:\s)*\,(?:\s)*viewlocaltable(?:\s)*\,(?:\s)*parameters(?:\s)*\,(?:\s)*attributes(?:\s)*,(?:\s)*globalresulttable\)",
        "await task._local(iternum, globalresulttable, parameters, attributes, db_objects,localtable, algorithm, viewlocaltable)",
        src,
    )
    src = src.split("\n", 1)[-1]
    src = (
        "async def dataflow(algorithm, parameters, attributes, db_objects, localtable, globaltable,  viewlocaltable, globalresulttable = None):\n"
        + src
        + "\nfunc[0] = dataflow"
    )
    #########################################################################################

    ast = parser.suite(src)
    exec(parser.compilest(ast))
    return await func[0](
        algorithm,
        parameters,
        attributes,
        db_objects,
        localtable,
        globaltable,
        viewlocaltable,
        globalresulttable,
    )


#### run function:
# creates unique table names
# get the algorithm name and accesses the corresponding python module
# creates the local views
# gets also db_objects: the connection objects to all the nodes of the federation
# initializes the connections between the servers and the remote/merge tables
# runs the dataflow
# cleans up the servers and returns the results


async def run(algorithm, params, db_objects):
    #### create unique table names
    table_id = get_uniquetablename()
    localtable = "local" + table_id
    globaltable = "global" + table_id
    viewlocaltable = "localview" + table_id
    globalresulttable = "globalres" + table_id
    result = []
    params = json.loads(params)

    ### get the corresponding algorithm python module using algorithm name
    module = get_package(algorithm)

    # create database views on local databases - each view processes the filters and the selected attributes on the requested table
    # the algorithm won't run directly on the local dataset but on the view
    await task.createlocalviews(db_objects, viewlocaltable, params)

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
                await transfer.initialize(
                    db_objects,
                    localtable,
                    globaltable,
                    globalresulttable,
                    algorithm_metadata["algorithms"][c]["local_schema"],
                    algorithm_metadata["algorithms"][c]["global_schema"],
                )
                #### initialize database tables
                await task.initialize(
                    db_objects,
                    localtable,
                    algorithm_metadata["algorithms"][c]["local_schema"],
                    globalresulttable,
                    algorithm_metadata["algorithms"][c]["global_schema"],
                )
                #### run the algorithm dataflow
                result = await dataflow(
                    module,
                    params["parameters"],
                    params["attributes"],
                    db_objects,
                    localtable,
                    globaltable,
                    viewlocaltable,
                    globalresulttable,
                )
            except:
<<<<<<< HEAD
                #### clean unused tables
                await task.clean_up(
                    db_objects,
                    globaltable,
                    localtable,
                    viewlocaltable,
                    globalresulttable,
                )
                raise
    ### clean up tables that are created during the execution
    await task.clean_up(
        db_objects, globaltable, localtable, viewlocaltable, globalresulttable
    )
    return result
=======
                 await task.clean_up(db_objects, globaltable, localtable, viewlocaltable, globalresulttable)
                 raise
      ### clean up tables that are created during the execution
      await task.clean_up(db_objects, globaltable, localtable, viewlocaltable, globalresulttable)
      return result
>>>>>>> d0e6fd68c503a1ef93f2fc217b32554ae583c213
