import functions


def setConnection(connection, functions, externalpath):
    try:
        connection.autocommit = True
    except:
        pass
    functions.register(connection, externalpath)
    functions.settings["stacktrace"] = True
    return connection


def connect(user, password, host, db, port, engine):
    Connection = functions.Connection()
    return functions.dbdialect.createConnection(
        Connection, user, password, host, db, port
    )


def connect_init(user, password, host, db, port, engine, path):
    functions.DIALECT = engine
    functions.imports(engine)
    functions.sqltransform.setimports(engine)
    Connection = functions.Connection()
    connection = functions.dbdialect.createConnection(
        Connection, user, password, host, db, port
    )
    connection = setConnection(connection, functions, path)
    return connection
