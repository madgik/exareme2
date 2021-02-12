# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.  If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright 1997 - July 2008 CWI, August 2008 - 2016 MonetDB B.V.

import asyncio
import logging
import platform

from aiopymonetdb import exceptions
from aiopymonetdb import mapi_async as mapi
from aiopymonetdb.sql import cursors

logger = logging.getLogger("pymonetdb")


class Connection(object):
    """A MonetDB SQL database connection"""

    default_cursor = cursors.Cursor

    def __init__(
        self,
        database,
        hostname=None,
        port=50000,
        username="monetdb",
        password="monetdb",
        unix_socket=None,
        autocommit=True,
        host=None,
        user=None,
        connect_timeout=-1,
    ):
        """Set up a connection to a MonetDB SQL database.

        args:
            database (str): name of the database
            hostname (str): Hostname where monetDB is running
            port (int): port to connect to (default: 50000)
            username (str): username for connection (default: "monetdb")
            password (str): password for connection (default: "monetdb")
            unix_socket (str): socket to connect to. used when hostname not set
                                (default: "/tmp/.s.monetdb.50000")
            autocommit (bool):  enable/disable auto commit (default: False)
            connect_timeout -- the socket timeout while connecting
                               (default: see python socket module)

        returns:
            Connection object

        """
        self.autocommit = autocommit
        self.sizeheader = True
        self.replysize = 0

        # The DB API spec is not specific about this
        if host:
            hostname = host
        if user:
            username = user

        if platform.system() == "Windows" and not hostname:
            hostname = "localhost"

        self.hostname = hostname
        self.port = int(port)
        self.username = username
        self.password = password
        self.database = database
        self.unixsocket = unix_socket
        self.connect_timeout = connect_timeout
        self.unix_socket = unix_socket
        self.autocommit = autocommit

    async def open(self):
        self.mapi = mapi.Connection()
        await self.mapi.connect(
            hostname=self.hostname,
            port=self.port,
            username=self.username,
            password=self.password,
            database=self.database,
            language="sql",
            unix_socket=self.unix_socket,
            connect_timeout=self.connect_timeout,
        )
        await self.set_sizeheader(True)
        await self.set_replysize(100)
        await self.set_autocommit(self.autocommit)

    def bind_str(self, parameter):
        return monetize.convert(parameter)

    async def close(self):
        """Close the connection.

        The connection will be unusable from this
        point forward; an Error exception will be raised if any operation
        is attempted with the connection. The same applies to all cursor
        objects trying to use the connection.  Note that closing a connection
        without committing the changes first will cause an implicit rollback
        to be performed.
        """
        if self.mapi:
            if not self.autocommit:
                self.rollback()
            await self.mapi.disconnect()
            self.mapi = None
        else:
            raise exceptions.Error("already closed")

    async def set_autocommit(self, autocommit):
        """
        Set auto commit on or off. 'autocommit' must be a boolean
        """
        await self.command("Xauto_commit %s" % int(autocommit))
        self.autocommit = autocommit

    async def set_sizeheader(self, sizeheader):
        """
        Set sizeheader on or off. When enabled monetdb will return
        the size a type. 'sizeheader' must be a boolean.
        """
        await self.command("Xsizeheader %s" % int(sizeheader))
        self.sizeheader = sizeheader

    async def set_replysize(self, replysize):
        await self.command("Xreply_size %s" % int(replysize))
        self.replysize = replysize

    async def commit(self):
        """
        Commit any pending transaction to the database. Note that
        if the database supports an auto-commit feature, this must
        be initially off. An interface method may be provided to
        turn it back on.

        Database modules that do not support transactions should
        implement this method with void functionality.
        """
        self.closed()
        return await self.cursor().execute("COMMIT")

    async def rollback(self):
        """
        This method is optional since not all databases provide
        transaction support.

        In case a database does provide transactions this method
        causes the database to roll back to the start of any
        pending transaction.  Closing a connection without
        committing the changes first will cause an implicit
        rollback to be performed.
        """
        self.closed()
        return await self.cursor().execute("ROLLBACK")

    ##################### added code to support federation #############################
    async def check_for_params(self, table, attributes):
        cur = cursors.Cursor(self)
        params = [table] + attributes
        attr = await cur.execute(
            "select columns.txt.name from tables,columns.txt where tables.id = columns.txt.table_id and tables.system = false and tables.name = %s and columns.txt.name in ("
            + ",".join(["%s" for x in set(attributes)])
            + ");",
            [(*params)],
        )

        if attr != len(attributes):
            res = await cur.fetchall()
            if res == []:
                raise Exception("Requested data.txt does not exist in all local nodes")
            raise Exception(
                "Attributes other than "
                + str(res)
                + " does not exist in all local nodes"
            )

    async def init_remote_connections(self, db_objects):
        await asyncio.sleep(0)

    async def broadcast_inparallel(
        self, local, globalresulttable, globalschema, dbname
    ):
        await local.cursor().execute(
            "DROP TABLE IF EXISTS  %s; CREATE REMOTE TABLE %s (%s) on 'mapi:%s';"
            % (globalresulttable, globalresulttable, globalschema, dbname)
        )


    async def merge(self, db_objects, localtable, globaltable, localschema):
        cur = cursors.Cursor(self)

        query = "DROP VIEW IF EXISTS " + globaltable + "; CREATE VIEW " + globaltable + " as "
        for i, local_node in enumerate(db_objects["local"]):
            await cur.execute(
                "DROP TABLE IF EXISTS %s_%s; CREATE REMOTE TABLE %s_%s (%s) on 'mapi:%s';"
                % (localtable, i, localtable, i, localschema, local_node["dbname"])
            )
            if i < len(db_objects["local"]) - 1:
                query += " select * from " + localtable + "_" + str(i) + " UNION ALL "
            else:
                query += " select * from " + localtable + "_" + str(i) + " ;"
        await cur.execute(query)


    async def merge1(self, db_objects, localtable, globaltable, localschema):
        cur = cursors.Cursor(self)
        await cur.execute("DROP TABLE IF EXISTS %s; CREATE MERGE TABLE %s (%s);" % (globaltable, globaltable, localschema))
        for i, local_node in enumerate(db_objects["local"]):
            await cur.execute(
                "DROP TABLE IF EXISTS %s_%s; CREATE REMOTE TABLE %s_%s (%s) on 'mapi:%s';"
                % (localtable, i, localtable, i, localschema, local_node["dbname"])
            )
            await cur.execute(
                "ALTER TABLE %s ADD TABLE %s_%s;" % (globaltable, localtable, i)
            )

    async def broadcast(self, db_objects, globalresulttable, globalschema):
        await asyncio.gather(
            *[
                self.broadcast_inparallel(
                    local_node["async_con"],
                    globalresulttable,
                    globalschema,
                    db_objects["global"]["dbname"],
                )
                for i, local_node in enumerate(db_objects["local"])
            ]
        )

    async def transferdirect(self, node1, localtable, node2, transferschema):
        await node2[2].cursor().execute(
            "CREATE REMOTE TABLE %s (%s) on 'mapi:%s';"
            % (localtable, transferschema, node1[1])
        )

    async def clean_tables(
        self, db_objects, globaltable, localtable, viewlocaltable, globalrestable
    ):
        try:
            await db_objects["global"]["async_con"].cursor().execute(
                "drop view if exists %s;" % globaltable
            )
            await db_objects["global"]["async_con"].cursor().execute(
                "drop table if exists %s;" % globalrestable
            )
            for i, local in enumerate(db_objects["local"]):
                await local["async_con"].cursor().execute(
                    "drop view if exists " + viewlocaltable + ";"
                )
                await local["async_con"].cursor().execute(
                    "drop table if exists " + globalrestable + ";"
                )
                await local["async_con"].cursor().execute(
                    "drop table if exists " + localtable + "_" + str(i) + ";"
                )
                await db_objects["global"]["async_con"].cursor().execute(
                    "drop table if exists " + localtable + "_" + str(i) + ";"
                )
        except:
            pass
    ##################### end of added code to support federation #############################

    def cursor(self):
        """
        Return a new Cursor Object using the connection.  If the
        database does not provide a direct cursor concept, the
        module will have to emulate cursors using other means to
        the extent needed by this specification.
        """
        return cursors.Cursor(self)

    async def execute(self, query):
        """ use this for executing SQL queries """
        return await self.command("s" + query + "\n;")

    async def command(self, command):
        """ use this function to send low level mapi commands """
        self.closed()
        return await self.mapi.cmd(command)

    def closed(self):
        """ check if there is a connection with a server """
        if not self.mapi:
            raise exceptions.Error("connection closed")
        return True

    def settimeout(self, timeout):
        """ set the amount of time before a connection times out """
        self.mapi.socket.settimeout(timeout)

    def gettimeout(self):
        """ get the amount of time before a connection times out """
        return self.mapi.socket.gettimeout()

    # these are required by the python DBAPI
    Warning = exceptions.Warning
    Error = exceptions.Error
    InterfaceError = exceptions.InterfaceError
    DatabaseError = exceptions.DatabaseError
    DataError = exceptions.DataError
    OperationalError = exceptions.OperationalError
    IntegrityError = exceptions.IntegrityError
    InternalError = exceptions.InternalError
    ProgrammingError = exceptions.ProgrammingError
    NotSupportedError = exceptions.NotSupportedError
