"""functions"""

VERSION = "1.0"
import ast
import inspect
import os
import os.path

import _cffi_backend
import lib
import lib.reimport
import pymonetdb
from _cffi_backend import _CDataBase
from cffi import FFI
from functions import aggregate
from functions import row
from functions import vtable

from . import setpath

DEBUG = False
RELOAD = False
results = {}
modules = {}
# try:
#    import apsw
# except:
#    import mspw
# monetdb specific
# import pymonetdb
import importlib
import time
import types

import functions

DIALECT = None
dbdialect = None
register_scalar = None
register_table = None
register_aggregate = None
getudfnames = None


## used to identify UDFs of type 3 and 6
def returns_multicolumn(func):
    try:
        source_code = inspect.getsource(func)
        tree = ast.parse(source_code)

        for node in ast.walk(tree):
            if isinstance(node, ast.Return):
                if isinstance(node.value, ast.Tuple):
                    return True
                else:
                    return False
        return False
    except:
        return False


def imports(dialect):
    """
    Dynamically import dialect modules and set global variables for the selected dialect.
    """
    global DIALECT, dbdialect, register_scalar, register_table, register_aggregate, getudfnames

    DIALECT = dialect

    # Get the directory of the current file and the dialects subfolder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dialects_dir = os.path.join(
        current_dir, "dialects"
    )  # Adjust to the dialects folder

    # Look for *_dialect.py files in the dialects directory
    dialect_modules = {
        file.split("_dialect.py")[0]: file.split(".py")[0]
        for file in os.listdir(dialects_dir)
        if file.endswith("_dialect.py")
    }

    if DIALECT not in dialect_modules:
        raise ImportError(
            f"Unsupported dialect: {DIALECT}. Available dialects: {list(dialect_modules.keys())}"
        )

    # Import the selected dialect module
    module_name = f"functions.dialects.{dialect_modules[DIALECT]}"
    dbdialect = importlib.import_module(module_name)

    # Dynamically import the required attributes from the selected module
    register_scalar = getattr(dbdialect, "register_scalar", None)
    register_table = getattr(dbdialect, "register_table", None)
    register_aggregate = getattr(dbdialect, "register_aggregate", None)
    getudfnames = getattr(dbdialect, "getudfnames", None)

    # Check if all required functions are present
    if any(
        attr is None
        for attr in [register_scalar, register_table, register_aggregate, getudfnames]
    ):
        raise ImportError(
            f"One or more required functions are missing from the {DIALECT} dialect."
        )


import copy
import logging
import re
import sys
import traceback
from typing import Generator
from typing import Iterator
from typing import Literal

from . import sqltransform

ffi = FFI()
sys.path.append(".")
sys.path.append("/home/johnfouf/arcade/")
extpath = None
functionspath = None
try:
    from collections import OrderedDict
except ImportError:
    # Python 2.6
    from lib.collections26 import OrderedDict

try:
    from inspect import isgeneratorfunction
except ImportError:
    # Python < 2.6
    def isgeneratorfunction(obj):
        return bool(
            (inspect.isfunction(object) or inspect.ismethod(object))
            and obj.__code__.co_flags & CO_GENERATOR
        )


alludfs = []
alludfs_pattern = ""
pattern = ""
patternfull = ""
patternalludfs = ""

# monetdb specific datatype map
# datatypemap = {
#    'STRING': 'char**',
#    'INT': 'int*',
#    'FLOAT': 'double*',
#    'BOOL': 'bool*',
#    'TINYINT': 'int8_t *',
#    'SMALLINT': 'short *',
#    'BIGINT': 'long long *',
#    'HUGEINT': '__int128 *',
# }

VTCREATE = "create temp table "
firstimport = True
test_connection = None
createfunctions = ""
udfs_cache = {}
statement_history = {}
settings = {
    "tracing": False,
    "vtdebug": False,
    "logging": False,
    "syspath": str(
        os.path.abspath(
            os.path.expandvars(os.path.expanduser(os.path.normcase(sys.path[0])))
        )
    ),
}

stfunctions = {"row": {}, "aggregate": {}, "vtable": {}}
dynfunctions = {"row": {}, "aggregate": {}, "vtable": {}}
functions = {"row": {}, "aggregate": {}, "vtable": {}}
multiset_functions = {}
functiontypes = {}
iterheader = "ITER" + chr(30)
alludfspattern = ""
variables = lambda _: _
variables.flowname = ""
variables.execdb = None
variables.filename = ""

privatevars = lambda _: _

rowfuncs = lambda _: _

oldexecdb = -1


def parse_set_debug(command):
    global DEBUG
    pattern = r"\b(?:select\s+)?set_debug\s*(?:\(\s*(.*?)\s*\)|\s+(.*?))?\s*;?\s*$"
    match = re.search(pattern, command, re.IGNORECASE)
    if match:
        arg = match.group(1) or match.group(2)
        if arg is None:  # No argument provided
            if DEBUG == False:
                DEBUG = True
                return "select set_debug(true);"
            elif DEBUG == True:
                DEBUG = False
                return "select set_debug(false);"
        # Trim and check the argument
        arg = arg.strip()
        if arg.lower() in ("1", "true"):
            DEBUG = True
            return "select set_debug(true);"
        elif arg.lower() in ("0", "false"):
            DEBUG = False
            return "select set_debug(false);"
        else:
            raise ValueError(
                f"YeSQL Error: Unexpected argument: {arg}, expected bool or no argument"
            )
    # Return None if no valid set_debug command is found
    return None


def parse_begin_transaction(command):
    pattern = re.compile(r"(?i)\bBEGIN\s+TRANSACTION\s*$", re.IGNORECASE)
    match = re.search(pattern, command)
    if match:
        return True
    return False


def parse_end_transaction(command):
    pattern = re.compile(r"(?i)COMMIT\s*$|;", re.IGNORECASE)
    match = re.search(pattern, command)
    if match:
        return True
    pattern = re.compile(r"(?i)ROLLBACK\s*$|;", re.IGNORECASE)
    match = re.search(pattern, command)
    if match:
        return True
    return False


def getvar(name):
    return variables.__dict__[name]


def setvar(name, value):
    variables.__dict__[name] = value


def mstr(s):
    if s == None:
        return None

    try:
        return str(s, "utf-8", errors="replace")
    except KeyboardInterrupt:
        raise
    except:
        # Parse exceptions that cannot be converted by unicode above
        try:
            return str(s)
        except KeyboardInterrupt:
            raise
        except:
            pass

    o = repr(s)
    if (o[0:2] == "u'" and o[-1] == "'") or (o[0:2] == 'u"' and o[-1] == '"'):
        o = o[2:-1]
    elif (o[0] == "'" and o[-1] == "'") or (o[0] == '"' and o[-1] == '"'):
        o = o[1:-1]
    o = o.replace("""\\n""", "\n")
    o = o.replace("""\\t""", "\t")
    return o


class YeSQLError(Exception):
    def __init__(self, msg):
        self.msg = mstr(msg)

    def __str__(self):
        merrormsg = "YeSQL SQLError: \n"
        if self.msg.startswith(merrormsg):
            return self.msg
        else:
            return merrormsg + self.msg


class OperatorError(YeSQLError):
    def __init__(self, opname, msg):
        self.msg = "Operator %s: %s" % (mstr(opname.upper()), mstr(msg))


class DynamicSchemaWithEmptyResultError(YeSQLError):
    def __init__(self, opname):
        self.msg = (
            "Operator %s: Cannot initialize dynamic schema virtual table without data"
            % (mstr(opname.upper()))
        )


def echofunctionmember(func):
    def wrapper(*args, **kw):
        if settings["tracing"]:
            if settings["logging"]:
                try:
                    lg = logging.LoggerAdapter(
                        logging.getLogger(__name__), {"flowname": variables.flowname}
                    )
                    if hasattr(lg.logger.parent.handlers[0], "baseFilename"):
                        lg.info(
                            "%s(%s)"
                            % (
                                func.__name__,
                                ",".join(
                                    list([repr(el) for el in args[1:]])
                                    + [
                                        "%s=%s" % (k, repr(v))
                                        for k, v in list(kw.items())
                                    ]
                                ),
                            )
                        )
                except Exception:
                    pass
            print(
                "%s(%s)"
                % (
                    func.__name__,
                    ",".join(
                        list(
                            [
                                repr(el)[:200] + ("" if len(repr(el)) <= 200 else "...")
                                for el in args[1:]
                            ]
                        )
                        + ["%s=%s" % (k, repr(v)) for k, v in list(kw.items())]
                    ),
                )
            )
        return func(*args, **kw)

    return wrapper


def iterwrapper(con, func, *args):
    global iterheader
    i = func(*args)
    si = bytes(iterheader + str(i), "utf8")
    con.openiters[si] = i
    return si.decode()


def iterwrapperaggr(con, func, self):
    global iterheader
    i = func(self)
    si = bytes(iterheader + str(i), "utf8")
    con.openiters[si] = i

    return si.decode()


def rec_build_sql(expression, rest):
    vt = re.search("(vt_\d+)", expression)
    if vt is not None:
        vt = vt.group()
        ast = re.sub("vt_", "table_", vt)
        for vtabs in rest[0]:
            if vt == vtabs[0]:
                expr = "(select * from " + vtabs[1] + "(" + vtabs[2] + ")) as " + ast
                expression = re.sub(vt, expr, expression)
                if re.search("(vt_\d+)", expression) is None:
                    return expression
                else:
                    return rec_build_sql(expression, rest)
    else:
        return expression


class Cursor(object):
    def __init__(self, w, connection):
        self.connection = connection
        self.__wrapped = w
        self.__vtables = []
        self.__permanentvtables = OrderedDict()
        self.__query = ""
        self.__initialised = True  # this should be last in init

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        return getattr(self.__wrapped, attr)

    def __setattr__(self, attr, value):
        if attr in self.__dict__:
            return object.__setattr__(self, attr, value)
        if (
            "_Cursor__initialised" not in self.__dict__
        ):  # this test allows attributes to be set in the __init__ method
            return object.__setattr__(self, attr, value)
        return setattr(self.__wrapped, attr, value)

    @echofunctionmember
    def executetrace(self, statements, bindings=None):
        start_time = time.time()
        # try:
        #    ret = self.__wrapped.execute('COMMIT;',bindings)
        # except:
        #    pass
        try:
            if statements != "":
                # print('statements: ', statements)
                ret = self.__wrapped.execute(statements, bindings)
                end_time = time.time()

                # Calculate the elapsed time
                elapsed_time = end_time - start_time

                # print(f"Elapsed execute time: {elapsed_time:.6f} seconds")
                return ret
            else:
                pass
        except KeyboardInterrupt:
            raise  # Optional: re-raise to propagate or allow caller to handle connection cleanup

        except Exception as e:
            try:  # avoid masking exception in recover statements
                raise e.with_traceback(sys.exc_info()[2])
            finally:
                pass

    def reloadfunctions(self):
        global pattern
        global alludfspattern
        global patternfull
        if not DEBUG:
            return
        # TODO add the possible new UDFs to the list for pattern for casts
        mods = findmodulespath(functionspath, "row", "functions")
        mods.extend(findmodulespath(functionspath, "aggregate", "functions"))
        mods.extend(findmodulespath(functionspath, "vtable", "functions"))
        mods.extend(findmodulespath("", extpath))
        newmodules = [x for x in mods if x not in modules]

        if len(newmodules) > 0:
            for module in newmodules:
                modlist = module.split(".")
                if len(modlist) == 2:
                    tmp = __import__(module)
                    register_ops(tmp.__dict__[modlist[1]], self)
                elif len(modlist) == 3:
                    tmp = __import__(module)
                    if modlist[1] == "row":
                        moddict = row.__dict__[modlist[2]]
                        register_ops(moddict, self)
                    if modlist[1] == "aggregate":
                        moddict = aggregate.__dict__[modlist[2]]
                        register_ops(moddict, self)
                    if modlist[1] == "vtable":
                        moddict = vtable.__dict__[modlist[2]]
                        register_ops(moddict, self)
        modified = lib.reimport.modified(newmodules)
        for r in newmodules:
            modules[r] = True
        if len(modified) == 0 or (modified == ["__main__"]):
            return
        try:
            for x in modified:
                if x != "__main__":
                    modlist = x.split(".")
                    if len(modlist) == 2:
                        tmp = __import__(x)
                        lib.reimport.reimport(x)
                        register_ops(tmp.__dict__[modlist[1]], self)
                    if len(modlist) == 3:
                        tmp = __import__(x)
                        if modlist[1] == "row":
                            lib.reimport.reimport(x)
                            moddict = row.__dict__[modlist[2]]
                            register_ops(moddict, self)
                        if modlist[1] == "aggregate":
                            lib.reimport.reimport(x)
                            moddict = aggregate.__dict__[modlist[2]]
                            register_ops(moddict, self)
                        if modlist[1] == "vtable":
                            lib.reimport.reimport(x)
                            moddict = vtable.__dict__[modlist[2]]
                            register_ops(moddict, self)
        except ValueError:
            pass
        alludfs = list(functions["vtable"].keys())
        alludfs_pattern = "|".join(r"\b{}\b".format(re.escape(x)) for x in alludfs)
        pattern = re.compile(
            rf"cast\s+'([a-zA-Z0-9,\s]+)'\s*({alludfs_pattern})",
            flags=re.VERBOSE | re.UNICODE | re.IGNORECASE,
        )
        alludfs.extend(
            x for x, func in functions["row"].items() if not isgeneratorfunction(func)
        )
        alludfs.extend(
            x
            for x, func in functions["aggregate"].items()
            if not isgeneratorfunction(func.final)
        )
        pattern1 = "|".join(r"\b{}\b".format(re.escape(x)) for x in alludfs)
        patternfull = re.compile(
            rf"cast\s*\(\s*({pattern1})\s*\(",
            flags=re.VERBOSE | re.UNICODE | re.IGNORECASE,
        )

    def execute(
        self,
        statements,
        jsonargs="",
        fromportal=False,
        bindings=None,
        parse=True,
        localbindings=None,
        postgres=None,
    ):  # overload execute statement
        global pattern, RELOAD, DEBUG
        q = parse_set_debug(statements)
        if q != None:
            return self.executetrace(q)
        if DEBUG:
            self.reloadfunctions()
        if localbindings != None:
            bindings = localbindings
        else:
            if bindings == None:
                bindings = variables.__dict__
            else:
                if type(bindings) is dict:
                    bindings.update(variables.__dict__)

        if not parse:
            self.__query = statements
            return self.executetrace(statements, bindings, parse)
        global alludfspattern

        # process casts
        def find_unbalanced_closing_parenthesis(s):
            stack = []
            single_quoted = False
            double_quoted = False

            for index, char in enumerate(s):
                if char == "'" and not double_quoted:
                    single_quoted = not single_quoted
                elif char == '"' and not single_quoted:
                    double_quoted = not double_quoted
                elif not single_quoted and not double_quoted:
                    if char == "(":
                        stack.append(index)
                    elif char == ")":
                        if not stack:
                            return index
                        stack.pop()

            if stack:
                return stack[
                    -1
                ]  # The index of the first unbalanced opening parenthesis
            return -1  # If no unbalanced closing parenthesis found

        originstatements = statements
        if statements in statement_history:
            statements = statement_history[statements]
        else:

            while True:
                res = pattern.search(statements)
                if res is None:
                    break
                funcdata = []
                funcdata.append(res.groups(0)[1])
                funcdata.append(res.groups(0)[0])
                statements = re.sub(
                    "cast\s+'([a-zA-Z0-9,\s]+)'\s*" + funcdata[0],
                    funcdata[0]
                    + "_"
                    + "_".join([x.strip().upper() for x in funcdata[1].split(",")]),
                    statements,
                    count=1,
                    flags=re.VERBOSE | re.UNICODE | re.IGNORECASE,
                )
                fedudf = False
                try:
                    if functions["vtable"][funcdata[0]].is_fedudf:
                        fedudf = True
                except:
                    pass
                if isgeneratorfunction(functions["vtable"][funcdata[0]]) or fedudf:
                    fedudf = False
                    newfuncname = (
                        funcdata[0]
                        + "_"
                        + "_".join([x.strip().lower() for x in funcdata[1].split(",")])
                    )
                    if newfuncname not in functions["vtable"]:
                        functions["vtable"][newfuncname] = functions["vtable"][
                            funcdata[0]
                        ]
                        functions["vtable"][newfuncname].returntype = [
                            x.strip().lower() for x in funcdata[1].split(",")
                        ]
                        # TODO this is a bug, returntype overwrites function object's previous casts
                        mtype = ""
                        datatypes = []
                        for num, tt in enumerate(
                            functions["vtable"][newfuncname].returntype
                        ):
                            if tt.lower() == "string":
                                mtype += (
                                    funcdata[0]
                                    + "_"
                                    + str(num)
                                    + "  "
                                    + dbdialect.db_sql_string
                                    + ","
                                )
                                datatypes.append(
                                    dbdialect.datatypemap[dbdialect.db_sql_string]
                                )
                            elif tt.lower() == "int":
                                mtype += (
                                    funcdata[0]
                                    + "_"
                                    + str(num)
                                    + "  "
                                    + dbdialect.db_sql_int
                                    + ","
                                )
                                datatypes.append(
                                    dbdialect.datatypemap[dbdialect.db_sql_int]
                                )
                            elif tt.lower() == "bigint":
                                mtype += (
                                    funcdata[0]
                                    + "_"
                                    + str(num)
                                    + "  "
                                    + dbdialect.db_sql_bigint
                                    + ","
                                )
                                datatypes.append(
                                    dbdialect.datatypemap[dbdialect.db_sql_bigint]
                                )
                            elif tt.lower() == "float":
                                mtype += (
                                    funcdata[0]
                                    + "_"
                                    + str(num)
                                    + "  "
                                    + dbdialect.db_sql_float
                                    + ","
                                )
                                datatypes.append(
                                    dbdialect.datatypemap[dbdialect.db_sql_float]
                                )
                        mtype = mtype.strip(",")
                        register_table(
                            newfuncname,
                            funcdata[0],
                            functions["vtable"][newfuncname],
                            mtype,
                            datatypes,
                            self,
                            Connection(),
                        )
                    statements = re.sub(
                        "cast\s*\(\s*" + funcdata[0],
                        newfuncname,
                        statements,
                        count=1,
                        flags=re.VERBOSE | re.UNICODE | re.IGNORECASE,
                    )

            # alludfs = list(functions['vtable'].keys())
            # alludfs.extend(x for x, func in functions['row'].items() if not isgeneratorfunction(func))
            # alludfs.extend(    x for x, func in functions['aggregate'].items() if not isgeneratorfunction(func.final))
            # pattern = '|'.join(r'\b{}\b'.format(re.escape(x)) for x in alludfs)
            # compiled_pattern = re.compile(rf'cast\s*\(\s*({pattern})\s*\(',flags=re.VERBOSE | re.UNICODE | re.IGNORECASE)

            while True:  # cast with standard syntax
                res = patternfull.search(statements)
                if res is None:
                    break
                funcdata = res.groups()
                result_index = find_unbalanced_closing_parenthesis(
                    statements[res.end() :]
                )
                datatype = re.search(
                    r"\s*as\s*((?:\s*[a-zA-Z0-9, _\s]+\s*)|(?:\([a-zA-Z0-9, _\s]+)\s*\))\s*\)",
                    statements[result_index + res.end() :],
                    flags=re.VERBOSE | re.UNICODE | re.IGNORECASE,
                )
                # if res2 is none raise syntax error
                funcdata += datatype.groups()
                fedudf = False
                try:
                    if functions["vtable"][funcdata[0]].is_fedudf:
                        fedudf = True
                except:
                    pass
                if funcdata[0] in functions["vtable"] and (
                    isgeneratorfunction(functions["vtable"][funcdata[0]]) or fedudf
                ):
                    fedudf = False
                    statements = (
                        statements[: result_index + res.end() + datatype.start()]
                        + statements[result_index + res.end() + datatype.end() :]
                    )
                    newfuncname = (
                        funcdata[0]
                        + "_"
                        + "_".join([x.strip().lower() for x in funcdata[1].split(",")])
                    )
                    if newfuncname not in functions["vtable"]:
                        functions["vtable"][newfuncname] = functions["vtable"][
                            funcdata[0]
                        ]
                        functions["vtable"][newfuncname].returntype = [
                            x.strip() for x in funcdata[1].split(",")
                        ]
                        mtype = ""
                        datatypes = []
                        for num, tt in enumerate(
                            functions["vtable"][newfuncname].returntype
                        ):
                            if tt.lower() == "string":
                                mtype += (
                                    funcdata[0]
                                    + "_"
                                    + str(num)
                                    + "  "
                                    + dbdialect.db_sql_string
                                    + ","
                                )
                                datatypes.append(
                                    dbdialect.datatypemap[dbdialect.db_sql_string]
                                )
                            elif tt.lower() == "int":
                                mtype += (
                                    funcdata[0]
                                    + "_"
                                    + str(num)
                                    + "  "
                                    + dbdialect.db_sql_int
                                    + ","
                                )
                                datatypes.append(
                                    dbdialect.datatypemap[dbdialect.db_sql_int]
                                )
                            elif tt.lower() == "bigint":
                                mtype += (
                                    funcdata[0]
                                    + "_"
                                    + str(num)
                                    + "  "
                                    + dbdialect.db_sql_bigint
                                    + ","
                                )
                                datatypes.append(
                                    dbdialect.datatypemap[dbdialect.db_sql_bigint]
                                )
                            elif tt.lower() == "float":
                                mtype += (
                                    funcdata[0]
                                    + "_"
                                    + str(num)
                                    + "  "
                                    + dbdialect.db_sql_float
                                    + ","
                                )
                                datatypes.append(
                                    dbdialect.datatypemap[dbdialect.db_sql_float]
                                )
                        mtype = mtype.strip(",")
                        register_table(
                            newfuncname,
                            funcdata[0],
                            functions["vtable"][newfuncname],
                            mtype,
                            datatypes,
                            self,
                            Connection(),
                        )
                    statements = re.sub(
                        "cast\s*\(\s*" + funcdata[0],
                        newfuncname,
                        statements,
                        count=1,
                        flags=re.VERBOSE | re.UNICODE | re.IGNORECASE,
                    )
                elif funcdata[0] in functions["row"]:
                    register_scalar(
                        funcdata[0],
                        functions["row"][funcdata[0]],
                        funcdata[1].upper(),
                        self,
                        Connection(),
                        True,
                    )
                    s = statements
                    s = (
                        s[: result_index + res.end() + datatype.start()]
                        + s[result_index + res.end() + datatype.end() :]
                    )
                    statements = re.sub(
                        "cast\s*\(\s*" + funcdata[0],
                        funcdata[0]
                        + "_"
                        + "_".join([x.upper() for x in funcdata[1].split(", ")]),
                        s,
                        count=1,
                        flags=re.VERBOSE | re.UNICODE | re.IGNORECASE,
                    )
                elif funcdata[0] in functions["aggregate"]:
                    register_aggregate(
                        funcdata[0],
                        functions["aggregate"][funcdata[0]],
                        funcdata[1].upper(),
                        self,
                        Connection(),
                        True,
                    )
                    s = statements
                    s = (
                        s[: result_index + res.end() + datatype.start()]
                        + s[result_index + res.end() + datatype.end() :]
                    )
                    statements = re.sub(
                        "cast\s*\(\s*" + funcdata[0],
                        funcdata[0]
                        + "_"
                        + "_".join([x.upper() for x in funcdata[1].split(", ")]),
                        s,
                        count=1,
                        flags=re.VERBOSE | re.UNICODE | re.IGNORECASE,
                    )
        statement_history[originstatements] = statements
        start_time = time.time()
        # TODO time goes here
        svts = sqltransform.transform(
            self,
            statements,
            multiset_functions,
            functions["vtable"],
            list(functions["row"].keys()),
            substitute=functions["row"]["subst"],
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        # print(f"Elapsed sql transform time: {elapsed_time:.6f} seconds")
        s = svts[0]
        try:
            if self.__vtables != []:
                self.executetrace(
                    "".join(["drop table " + x + ";" for x in reversed(self.__vtables)])
                )
                self.__vtables = []
            for i in svts[1]:
                createvirtualsql = None
                if re.match(r"\s*$", i[2]) is None:
                    sep = ","
                else:
                    sep = ""
                createvirtualsql = "select * from " + i[1] + "(" + i[2] + ") ;"
                # try:
                #        self.executetrace(createvirtualsql)
                # except Exception as e:
                #    strex = mstr(e)
                #    #self.executetrace("drop table if exists "+str(svts[1][0][0])+";")
                #    raise e.with_traceback(sys.exc_info()[2])

                # if len(i) == 4:
                #    self.__permanentvtables[i[0]] = createvirtualsql
                # else:
                #    self.__vtables.append(i[0])
            self.__query = s
            ### here goes fusion code -
            if 1 == 0:
                try:
                    results = {}
                    if fromportal == True and jsonargs != "":
                        results = qfusor.execute_query(
                            self, self.__query, jsonargs, postgres
                        )
                        return results
                    if fromportal == False and jsonargs == "":
                        jsonargs = (
                            '{"sqlQuery":"'
                            + self.__query
                            + '", "nesting":7,"scalar":true,"aggregate":true,"table":true,"udfrel":true,"udf_reorder":true,"engine":"MonetDB","enginespec":false}'
                        )
                        results = qfusor.execute_query(
                            self, self.__query, jsonargs, postgres
                        )
                    s = results["fusedquery"]
                    print("fused: ", s)
                    self.__query = s
                except:
                    raise
            """
                1. execute set optimizer='fusion_pipe'
                2. run explain + query -> this result can return to the portal -> produces files in /tmp (fused.sql, fused.py, code_fused.py, fused.h, graph.dot, fusedGraph.dot)
                3. compile (pypy3 fused.py)
                4. Link (export LD_LIBRARY_PATH=/tmp)
                5. Change optimizer to default
                6. Run fused.sql from mclient
                """

            """
                final steps:
                0. set optimizer='fusion_pipe'
                1. if query string includes a UDF, explain self.__query
                2. parse explain and produce dependency graph
                3. select fused components and submit appropriate create function statements
                4. rewrite self.__query and submit for execution

                """
            return self.executetrace(s, bindings)
        except Exception as e:
            if settings["tracing"]:
                traceback.print_exc(limit=sys.getrecursionlimit())
            try:  # avoid masking exception in recover statements
                raise e.with_traceback(sys.exc_info()[2])
            finally:
                try:
                    self.cleanupvts()
                except:
                    pass

    def getdescriptionsafe(self):
        return self.description

    def close(self, force=False):
        self.cleanupvts()
        return self.__wrapped.close()

    def cleanupvts(self):
        if self.__vtables != []:
            for t in reversed(self.__vtables):
                pass
                # self.executetrace('drop table if exists ' + t)
                # self.executetrace("drop function if exists expand_"+t+";")
            self.__vtables = []


def Connection():
    """
    Factory function to create the Connection class dynamically.
    """
    if dbdialect is None:
        raise RuntimeError(
            "Dialect is not set. Call 'set_dialect' before using Connection."
        )

    # Dynamically create the Connection class
    class Connection(dbdialect.dbConnection):
        def cursor(self):
            return Cursor(dbdialect.dbConnection.cursor(self), self)

    return Connection
    # print('fifi')
    # print(Connection)
    # print('fofo')


def register(connection=None, externalpath=None):
    global firstimport, oldexecdb, extpath, functionspath, DEBUG, pattern, patternfull
    if connection == None:
        connection = dbdialect.dbconnect(
            username=user, password=password, hostname=host, database=db, port=port
        )
    connection.openiters = {}
    connection.registered = True

    ccur = connection.cursor()
    funcexpand = dbdialect.funcexpand
    funcmytable = dbdialect.funcmytable

    # monetdb specific retrieve udf names/types
    dbfunctions = {"row": {}, "aggregate": {}, "vtable": {}}
    ccur = getudfnames(ccur, dbfunctions)
    ccur.executetrace(funcexpand)
    ccur.executetrace(funcmytable)
    ccur.close()
    # To avoid db corruption set connection to fullfsync mode when MacOS is detected

    functionspath = os.path.abspath(__path__[0])

    ## Register main functions of YeSQL (functions)
    rowfiles = findmodules(functionspath, "row")
    for f in rowfiles:
        modules["functions.row." + f] = True
    aggrfiles = findmodules(functionspath, "aggregate")
    for f in aggrfiles:
        modules["functions.aggregate." + f] = True
    vtabfiles = findmodules(functionspath, "vtable")
    for f in vtabfiles:
        modules["functions.vtable." + f] = True
    [__import__("functions.row" + "." + module) for module in rowfiles]
    [__import__("functions.aggregate" + "." + module) for module in aggrfiles]
    [__import__("functions.vtable" + "." + module) for module in vtabfiles]

    # Register aggregate functions
    for module in aggrfiles:
        moddict = aggregate.__dict__[module]
        register_ops(moddict, connection)

    # Register row functions
    for module in rowfiles:
        moddict = row.__dict__[module]
        register_ops(moddict, connection)

    for module in vtabfiles:
        moddict = vtable.__dict__[module]
        register_ops(moddict, connection)

    if externalpath is not None:
        extpath = externalpath
        externalfiles = findmodules(externalpath, "")
        for f in externalfiles:
            modules[externalpath + "." + f] = True
        sys.path.append(externalpath)
        expath = os.path.basename(os.path.normpath(externalpath))
        expathmod = __import__(expath)
        for module in externalfiles:
            tmp = __import__(expath + "." + module)
            register_ops(tmp.__dict__[module], connection)

    ## Register YeSQL local functions (functionslocal)
    functionslocalpath = os.path.abspath(
        os.path.join(functionspath, "..", "functionslocal")
    )

    flrowfiles = findmodules(functionslocalpath, "row")
    flaggrfiles = findmodules(functionslocalpath, "aggregate")
    flvtabfiles = findmodules(functionslocalpath, "vtable")

    for module in flrowfiles:
        tmp = __import__("functionslocal.row." + module)
        register_ops(tmp.row.__dict__[module], connection)

    for module in flaggrfiles:
        tmp = __import__("functionslocal.aggregate." + module)
        register_ops(tmp.aggregate.__dict__[module], connection)

    localvtable = lambda x: x
    for module in flvtabfiles:
        localvtable.__dict__[module] = __import__(
            "functionslocal.vtable." + module, fromlist=["functionslocal.vtable"]
        )

    if len(flvtabfiles) != 0:
        register_ops(localvtable, connection)

    ## Register db local functions (functions in db path)
    if variables.execdb != oldexecdb:
        oldexecdb = variables.execdb
        dbpath = None

        if variables.execdb != None:
            dbpath = os.path.join(
                os.path.abspath(os.path.dirname(variables.execdb)), "functions"
            )

        if dbpath == None or not os.path.exists(dbpath):
            currentpath = os.path.abspath(
                os.path.join(os.path.abspath("."), "functions")
            )
            if os.path.exists(currentpath):
                dbpath = currentpath

        if dbpath != None and os.path.exists(dbpath):
            if os.path.abspath(dbpath) != os.path.abspath(functionspath):

                sys.path.append(dbpath)

                if os.path.exists(os.path.join(dbpath, "row")):
                    lrowfiles = findmodules(dbpath, "row")
                    sys.path.append(
                        (os.path.abspath(os.path.join(os.path.join(dbpath), "row")))
                    )
                    for module in lrowfiles:
                        tmp = __import__(module)
                        register_ops(tmp, connection)

                if os.path.exists(os.path.join(dbpath, "aggregate")):
                    sys.path.append(
                        (
                            os.path.abspath(
                                os.path.join(os.path.join(dbpath), "aggregate")
                            )
                        )
                    )
                    laggrfiles = findmodules(dbpath, "aggregate")
                    for module in laggrfiles:
                        tmp = __import__(module)
                        register_ops(tmp, connection)

                if os.path.exists(os.path.join(dbpath, "vtable")):
                    sys.path.append(
                        (os.path.abspath(os.path.join(os.path.join(dbpath), "vtable")))
                    )
                    lvtabfiles = findmodules(dbpath, "vtable")
                    tmp = lambda x: x
                    for module in lvtabfiles:
                        tmp.__dict__[module] = __import__(module)

                    if localvtable != None:
                        register_ops(tmp, connection)
    # register_all(connection)
    alludfs = list(functions["vtable"].keys())
    alludfs_pattern = "|".join(r"\b{}\b".format(re.escape(x)) for x in alludfs)
    pattern = re.compile(
        rf"cast\s+'([a-zA-Z0-9,\s]+)'\s*({alludfs_pattern})",
        flags=re.VERBOSE | re.UNICODE | re.IGNORECASE,
    )
    alludfs.extend(
        x for x, func in functions["row"].items() if not isgeneratorfunction(func)
    )
    alludfs.extend(
        x
        for x, func in functions["aggregate"].items()
        if not isgeneratorfunction(func.final)
    )
    pattern1 = "|".join(r"\b{}\b".format(re.escape(x)) for x in alludfs)
    patternfull = re.compile(
        rf"cast\s*\(\s*({pattern1})\s*\(", flags=re.VERBOSE | re.UNICODE | re.IGNORECASE
    )
    try:
        ccur = connection.cursor()
        ccur.executetrace("select get_debug();")
        d = ccur.next()[0]
        if d == 0:
            DEBUG = False
        elif d == 1:
            DEBUG = True
    except:
        pass
    firstimport = False


def register_ops(module, connection):
    global rowfuncs, firstimport

    def opexists(op):
        if firstimport:
            return (
                op in functions["vtable"]
                or op in functions["row"]
                or op in functions["aggregate"]
            )
        else:
            return False

    def wrapfunction(con, opfun):
        return lambda *args: iterwrapper(con, opfun, *args)

    def wrapaggr(con, opfun):
        return lambda self: iterwrapperaggr(con, opfun, self)

    def wrapaggregatefactory(wlambda):
        return lambda cls: (cls(), cls.step, wlambda)

    def check_annotations(fobject):
        try:
            annotations = [
                annotation
                for _, annotation in fobject.__annotations__.items()
                if _ != "return"
            ]
        except:
            return False
        cdata_class = annotations[0] if annotations else None
        return (
            type(fobject).__name__ == "function"
            and isgeneratorfunction(fobject)
            and any(
                isinstance(annotation, Generator)
                or isinstance(annotation, Iterator)
                or type(annotation) is cdata_class
                for annotation in annotations
            )
        )

    for f in module.__dict__:
        fobject = module.__dict__[f]
        if (
            hasattr(fobject, "registered")
            and type(fobject.registered).__name__ == "bool"
            and fobject.registered == True
        ):
            opname = f.lower()
            if firstimport:
                if opname != f:
                    raise YeSQLError(
                        "Extended SQLERROR: Function '"
                        + module.__name__
                        + "."
                        + f
                        + "' uses uppercase characters. Functions should be lowercase"
                    )

                if opname.upper() in sqltransform.sqlparse.keywords.KEYWORDS:
                    raise YeSQLError(
                        "Extended SQLERROR: Function '"
                        + module.__name__
                        + "."
                        + opname
                        + "' is a reserved SQL function"
                    )

            if type(fobject).__name__ == "module":
                # if opexists(opname):
                #    raise YeSQLError("Extended SQLERROR: Vtable '"+opname+"' name collision with other operator")
                functions["vtable"][opname] = fobject
                modinstance = fobject.Source()
                modinstance._YeSQLVT = True
            # connection.createmodule(opname, modinstance)

            value_to_check = "<class '_cffi_backend._CDataBase'>"
            value_to_check1 = "typing.Generator"
            value_to_check2 = "typing.Iterator"
            value_to_check3 = "<class 'numpy.ndarray'>"
            # if check_annotations(fobject):
            checklist = []
            try:
                checklist = [
                    str(annotation)
                    for parameter, annotation in fobject.__annotations__.items()
                    if parameter != "return"
                ]
            except:
                pass
            fedudf = False
            try:
                if fobject.is_fedudf:
                    fedudf = True
            except:
                pass
            if (hasattr(fobject, "table") and fobject.table == True) or (
                type(fobject).__name__ == "function"
                and isgeneratorfunction(fobject)
                and (
                    value_to_check2 in checklist
                    or value_to_check1 in checklist
                    or value_to_check in checklist
                    or value_to_check3 in checklist
                )
            ):
                fedudf = False
                functions["vtable"][opname] = fobject
                functiontypes[opname] = 5
                annot = fobject.__annotations__
                fobject = wrapfunction(connection, fobject)
                fobject.multiset = True
                fobject.annotations = annot

                staticudf = 0
                try:
                    return_type = fobject.annotations["return"].__args__
                    staticudf = 1
                except:
                    dynfunctions["vtable"]["opname"] = fobject
                    ### cast all dynamic functions to work with string output
                    ### user defined command cast may change their output
                    ### TODO: change this all dynamic table functions return output schema = input schema
                    # register_table(opname, opname, functions['vtable'][opname], 'STRING', 'STRING',connection)
                mtype = ""
                datatypes = []

                if staticudf:
                    stfunctions["vtable"][opname] = fobject
                    # monetdb specific terms (STRING INT DOUBLE)
                    for num, tt in enumerate(return_type):
                        if tt is str:
                            mtype += (
                                opname
                                + "_"
                                + str(num)
                                + "  "
                                + dbdialect.db_sql_string
                                + ","
                            )
                            datatypes.append(
                                dbdialect.datatypemap[dbdialect.db_sql_string]
                            )
                        elif tt is int:
                            mtype += (
                                opname
                                + "_"
                                + str(num)
                                + "  "
                                + dbdialect.db_sql_int
                                + ","
                            )
                            datatypes.append(
                                dbdialect.datatypemap[dbdialect.db_sql_int]
                            )
                        elif isinstance(tt, types.FunctionType):
                            mtype += (
                                opname
                                + "_"
                                + str(num)
                                + "  "
                                + dbdialect.db_sql_bigint
                                + ","
                            )
                            datatypes.append(
                                dbdialect.datatypemap[dbdialect.db_sql_bigint]
                            )
                        elif tt is float:
                            mtype += (
                                opname
                                + "_"
                                + str(num)
                                + "  "
                                + dbdialect.db_sql_float
                                + ","
                            )
                            datatypes.append(
                                dbdialect.datatypemap[dbdialect.db_sql_float]
                            )
                        elif tt is Literal:
                            mtype += (
                                opname
                                + "_"
                                + str(num)
                                + "  "
                                + dbdialect.db_sql_string
                                + ","
                            )
                            datatypes.append(
                                dbdialect.datatypemap[dbdialect.db_sql_string]
                            )
                    mtype = mtype.strip(",")
                    stfunctions["vtable"][opname].datatype = datatypes
                    register_table(
                        opname,
                        opname,
                        functions["vtable"][opname],
                        mtype,
                        datatypes,
                        connection,
                        Connection(),
                    )

            elif type(fobject).__name__ == "function":
                functions["row"][opname] = fobject
                if isgeneratorfunction(fobject):
                    # set type to type 4
                    functiontypes[opname] = 4
                    annot = fobject.__annotations__
                    fobject = wrapfunction(connection, fobject)
                    fobject.multiset = True
                    fobject.annotations = annot
                elif returns_multicolumn(fobject):
                    functiontypes[opname] = 3
                    ### set type to type 3
                else:
                    functiontypes[opname] = 1
                setattr(rowfuncs, opname, fobject)
                staticudf = 0
                try:
                    if fobject.multiset == True:
                        register_scalar(
                            opname,
                            functions["row"][opname],
                            "STRING",
                            connection,
                            Connection(),
                        )
                except:
                    pass
                try:
                    return_type = fobject.__annotations__["return"]
                    staticudf = 1
                except:
                    dynfunctions["row"]["opname"] = fobject
                    ### cast all dynamic functions to work with string output
                    ### user defined command cast may change their output
                    register_scalar(
                        opname,
                        functions["row"][opname],
                        "STRING",
                        connection,
                        Connection(),
                    )
                mtype = None
                if staticudf:
                    stfunctions["row"][opname] = fobject
                    if return_type is str:
                        mtype = "STRING"
                    elif return_type is int:
                        mtype = "INT"
                    elif return_type is float:
                        mtype = "float"
                    elif return_type is Literal:
                        mtype = "STRINGLITERAL"
                    try:
                        if fobject.multiset:
                            pass
                    except:
                        register_scalar(
                            opname,
                            functions["row"][opname],
                            mtype,
                            connection,
                            Connection(),
                        )

            elif type(fobject).__name__ == "type":

                #  if opexists(opname):
                #      raise YeSQLError("Extended SQLERROR: Aggregate operator '"+module.__name__+'.'+opname+"' name collision with other operator")
                functions["aggregate"][opname] = fobject

                if isgeneratorfunction(fobject.final):
                    # ### set type to type 7
                    functiontypes[opname] = 7
                    wlambda = wrapaggr(connection, fobject.final)
                    fobject.multiset = True
                    setattr(
                        fobject, "factory", classmethod(wrapaggregatefactory(wlambda))
                    )
                    register_aggregate(
                        opname,
                        functions["aggregate"][opname],
                        "FLOAT",
                        connection,
                        Connection(),
                    )
                # connection.createaggregatefunction(opname, fobject.factory)
                else:
                    if returns_multicolumn(fobject.final):
                        functiontypes[opname] = 6
                    else:
                        functiontypes[opname] = 2
                    ### set type to type 6
                    try:
                        return_type = fobject.final.__annotations__["return"]
                        if return_type is str:
                            mtype = "STRING"
                        elif return_type is int:
                            mtype = "INT"
                        elif return_type is float:
                            mtype = "FLOAT"
                        setattr(
                            fobject,
                            "factory",
                            classmethod(lambda cls: (cls(), cls.step, cls.final)),
                        )
                        register_aggregate(
                            opname,
                            functions["aggregate"][opname],
                            mtype,
                            connection,
                            Connection(),
                        )
                    except:
                        setattr(
                            fobject,
                            "factory",
                            classmethod(lambda cls: (cls(), cls.step, cls.final)),
                        )
                        register_aggregate(
                            opname,
                            functions["aggregate"][opname],
                            "FLOAT",
                            connection,
                            Connection(),
                        )
                # connection.createaggregatefunction(opname, fobject.factory)

            try:
                if fobject.multiset:
                    try:
                        multiset_functions[opname] = fobject.annotations["return"]
                    except:
                        try:
                            multiset_functions[opname] = fobject.final.__annotations__[
                                "return"
                            ]
                        except:
                            multiset_functions[opname] = True
            except:
                pass


def register_all(connection):
    global createfunctions
    if isinstance(connection, Connection):
        connection.cursor().execute(createfunctions)
    else:
        connection.execute(createfunctions)
    createfunctions = ""


def testfunction():
    global test_connection, settings

    test_connection = Connection(":memory:")
    register(test_connection)
    variables.execdb = ":memory:"


def settestdb(testdb):
    global test_connection, settings

    abstestdb = str(
        os.path.abspath(
            os.path.expandvars(os.path.expanduser(os.path.normcase(testdb)))
        )
    )
    test_connection = Connection(abstestdb)
    register(test_connection)
    variables.execdb = abstestdb


def sql(sqlquery):
    import locale

    from lib import pptable

    global test_connection

    language, output_encoding = locale.getdefaultlocale()

    if output_encoding == None:
        output_encoding = "UTF8"

    test_cursor = test_connection.cursor()

    e = test_cursor.execute(sqlquery.decode(output_encoding))
    # try:
    desc = test_cursor.getdescription()
    print(
        pptable.indent([[x[0] for x in desc]] + [x for x in e], hasHeader=True), end=" "
    )
    # except apsw.ExecutionCompleteError:
    #    print('', end=' ')
    test_cursor.close()


def table(tab, num=""):
    import shlex

    """
    Creates a test table named "table". It's columns are fitted to the data
    given to it and are automatically named a, b, c, ...

    'num' parameter:
    If a 'num' parameter is given then the table will be named for example
    table1 when num=1, table2 when num=2 ...

    Example:

    table('''
    1   2   3
    4   5   6
    ''')

    will create a table named 'table' having the following data:

    a   b   c
    ---------
    1   2   3
    4   5   6

    """

    colnames = "abcdefghijklmnop"
    import re

    tab = tab.splitlines()
    tab = [re.sub(r"[\s\t]+", " ", x.strip()) for x in tab]
    tab = [x for x in tab if x != ""]
    # Convert NULL to None
    tab = [[(y if y != "NULL" else None) for y in shlex.split(x)] for x in tab]

    numberofcols = len(tab[0])

    if num == "":
        num = "0"

    createsql = "create table table" + str(num) + "("
    insertsql = "insert into table" + str(num) + " values("
    for i in range(0, numberofcols):
        createsql = createsql + colnames[i] + " str" + ","
        insertsql = insertsql + "?,"

    createsql = createsql[0:-1] + ")"
    insertsql = insertsql[0:-1] + ")"

    test_cursor = test_connection.cursor()
    try:
        test_cursor.execute(createsql)
    except:
        test_cursor.execute("drop table table" + str(num))
        test_cursor.execute(createsql)

    test_cursor.executemany(insertsql, tab)


def findmodules(abspath, relativepath):
    return [
        os.path.splitext(file)[0]
        for file in os.listdir(os.path.join(abspath, relativepath))
        if file.endswith(".py") and not file.startswith("_")
    ]


def findmodulespath(abspath, relativepath, firstpath=None):
    if firstpath == None:
        return [
            relativepath + "." + os.path.splitext(file)[0]
            for file in os.listdir(os.path.join(abspath, relativepath))
            if file.endswith(".py") and not file.startswith("_")
        ]
    else:
        return [
            firstpath + "." + relativepath + "." + os.path.splitext(file)[0]
            for file in os.listdir(os.path.join(abspath, relativepath))
            if file.endswith(".py") and not file.startswith("_")
        ]


def table1(tab):
    table(tab, num=1)


def table2(tab):
    table(tab, num=2)


def table3(tab):
    table(tab, num=3)


def table4(tab):
    table(tab, num=4)


def table5(tab):
    table(tab, num=5)


def table6(tab):
    table(tab, num=6)


def setlogfile(file):
    pass
