import logging
import sys

import tornado.web
from tornado.log import enable_pretty_logging
from tornado.options import define, options

import connections
import run_algorithm
import json
WEB_SERVER_PORT = 9999#7779
define("port", default=WEB_SERVER_PORT, help="run on the given port", type=int)


class AlgorithmException(Exception):
    def __init__(self, message):
        super(AlgorithmException, self).__init__(message)


class Application(tornado.web.Application):
    def __init__(self):
        handlers = [(r"/", MainHandler)]
        tornado.web.Application.__init__(self, handlers)


class BaseHandler(tornado.web.RequestHandler):
    def __init__(self, *args):
        tornado.web.RequestHandler.__init__(self, *args)


class MainHandler(BaseHandler):
    # logging stuff..
    enable_pretty_logging()
    logger = logging.getLogger("MainHandler")
    hdlr = logging.FileHandler("./mserver.log", "w+")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.DEBUG)

    access_log = logging.getLogger("tornado.access")
    app_log = logging.getLogger("tornado.application")
    gen_log = logging.getLogger("tornado.general")
    access_log.addHandler(hdlr)
    app_log.addHandler(hdlr)
    gen_log.addHandler(hdlr)
    dbs = connections.Connections()

    async def post(self):
        json_str = self.request.body
        json_str = json_str.decode('utf8').replace("'", '"')

        data=json.loads(json_str)

        algorithmParams = data["algorithmParams"]
        dataParams = data["dataParams"]
        print(f"(mserver::post) \nalgorithmParams->{algorithmParams} \ndataParams->{dataParams}\n")

        #### new connection per request - required since connection objects are not thread safe at the time
        await self.dbs.initialize()
        db_objects = await self.dbs.acquire()

        try:
            result = await run_algorithm.run(algorithmParams, dataParams, db_objects)
            self.write(f"ALGORITHM RESULT -> {result}")

        except Exception as e:
            # raise tornado.web.HTTPError(status_code=500,log_message="...the log message??")
            self.logger.debug(
                "(MadisServer::post) QueryExecutionException: {}".format(str(e))
            )
            await self.dbs.release(db_objects)
            self.write("Error: " + str(e))
            self.finish()
            raise

        await self.dbs.release(db_objects)
        self.logger.debug("(mserver.py::MadisServer::post) ALGORITHM RESULT -> {}".format(result))
        self.finish()

class PostParameters:
    def __init__(self,algorithmParams,dataParams):
        self.algorithmParams=algorithmParams
        self.dataParams=dataParams

    def __repr__(self):
        return f"<algorithmParams:{self.algorithmParams} dataParams->{dataParams}>"# functions->{self.functions}>"


def main(args):
    sockets = tornado.netutil.bind_sockets(options.port)
    server = tornado.httpserver.HTTPServer(Application())
    server.add_sockets(sockets)
    tornado.ioloop.IOLoop.instance().start()


if __name__ == "__main__":
    main(sys.argv)
