import algorithms
import json
import asyncio
import time

current_time = lambda: int(round(time.time() * 1000))


def bindparameters(parameters):
    bindparams = []
    for i in parameters:
        if isinstance(i, (int, float, complex)):
            bindparams.append(i)
        else:
            bindparams.append(db_objects["global"]["async_con"].bind_str(i))
    return bindparams


async def local_execute(local, localtable, id, sqlscript):
    query = (
        "delete from "
        + localtable
        + "_"
        + str(id)
        + "; insert into "
        + localtable
        + "_"
        + str(id)
        + " "
        + sqlscript
    )
    await local.cursor().execute(query)


async def create_view(local, viewlocaltable, attributes, table, filterpart, vals):
    if filterpart == " ":
        query = (
            "CREATE VIEW "
            + viewlocaltable
            + " AS select "
            + ",".join(attributes)
            + " from "
            + table
            + ";"
        )
        await local.cursor().execute(query)
    else:
        query = (
            "CREATE VIEW "
            + viewlocaltable
            + " AS select "
            + ",".join(attributes)
            + " from "
            + table
            + " where "
            + filterpart
            + ";"
        )
        await local.cursor().execute(query, vals)


async def createlocalviews(db_objects, viewlocaltable, params):
    t1 = current_time()

    table = params["table"]
    attributes = []
    for attribute in params["attributes"]:
        attributes.append(attribute)
    for formula in params["filters"]:
        for attribute in formula:
            if attribute[0] not in attributes:
                attributes.append(attribute[0])
    check_for_params_calls = [
        local["async_con"].check_for_params(table, attributes)
        for i, local in enumerate(db_objects["local"])
    ]
    await asyncio.gather(*check_for_params_calls)

    filterpart = " "
    vals = []
    for j, formula in enumerate(params["filters"]):
        andpart = " "
        for i, filt in enumerate(formula):
            if filt[1] not in [">", "<", "<>", ">=", "<=", "="]:
                raise Exception("Operator " + filt[1] + " not valid")
            andpart += filt[0] + filt[1] + "%s"
            vals.append(filt[2])
            if i < len(formula) - 1:
                andpart += " and "
        if andpart != " ":
            filterpart += "(" + andpart + ")"
        if j < len(params["filters"]) - 1:
            filterpart += " or "

    create_view_calls = [
        create_view(
            local["async_con"],
            viewlocaltable,
            params["attributes"],
            params["table"],
            filterpart,
            vals,
        )
        for local in db_objects["local"]
    ]
    await asyncio.gather(*create_view_calls)
    print("time " + str(current_time() - t1))


async def initialize(
    db_objects, localtable, localschema, globalresulttable, globalschema
):
    create_query = "create table if not exists %s (%s);" % (
        globalresulttable,
        globalschema,
    )
    await db_objects["global"]["async_con"].cursor().execute(
        "create table if not exists %s (%s);" % (globalresulttable, globalschema)
    )
    for i, local in enumerate(db_objects["local"]):
        query = "create table %s (%s);" % (localtable + "_" + str(i), localschema)
        await local["async_con"].cursor().execute(query)


async def _local(
    iternum,
    globalresulttable,
    parameters,
    attr,
    db_objects,
    localtable,
    algorithm,
    viewlocaltable,
):
    t1 = current_time()
    sqlscript = algorithm._local(
        iternum, viewlocaltable, bindparameters(parameters), attr, globalresulttable
    )
    local_execute_calls = [
        local_execute(local["async_con"], localtable, id, sqlscript)
        for id, local in enumerate(db_objects["local"])
    ]
    await asyncio.gather(*local_execute_calls)
    print("time " + str(current_time() - t1))


async def _global(
    iternum,
    globaltable,
    parameters,
    attr,
    db_objects,
    localtable,
    globalresulttable,
    algorithm,
    viewlocaltable,
):
    t1 = current_time()
    sqlscript = algorithm._global(
        iternum, globaltable, bindparameters(parameters), attr
    )
    query = (
        "delete from "
        + globalresulttable
        + "; insert into "
        + globalresulttable
        + " "
        + sqlscript
    )
    await db_objects["global"]["async_con"].cursor().execute(query)
    cur = db_objects["global"]["async_con"].cursor()
    result = await cur.execute("select * from %s;" % globalresulttable)
    print("time " + str(current_time() - t1))
    return cur.fetchall()


async def clean_up(db_objects, globaltable, localtable, viewlocaltable, globalrestable):
    await db_objects["global"]["async_con"].clean_tables(
        db_objects, globaltable, localtable, viewlocaltable, globalrestable
    )
