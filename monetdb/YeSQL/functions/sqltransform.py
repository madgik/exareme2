# coding: utf-8
import importlib
import os
import time

import sqlparse.sql
from functions import DIALECT

from . import setpath

parseplan = None
register_expand = None
plandatatypes = None
datatypemap = None
udftypes = None


def setimports(dbdialect):
    """
    Dynamically import functions and attributes from the selected dialect module.
    """
    global DIALECT, parseplan, register_expand, plandatatypes, datatypemap, udftypes
    DIALECT = dbdialect
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
    parseplan = getattr(dbdialect, "parseplan", None)
    register_expand = getattr(dbdialect, "register_expand", None)
    plandatatypes = getattr(dbdialect, "plandatatypes", None)
    datatypemap = getattr(dbdialect, "datatypemap", None)
    udftypes = getattr(dbdialect, "udftypes", None)
    # Check if all required functions are present
    if any(
        attr is None
        for attr in [parseplan, register_expand, plandatatypes, datatypemap]
    ):
        raise ImportError(
            f"One or more required functions are missing from the {DIALECT} dialect."
        )


import hashlib
import json
import re
import zlib

import functions
import sqlparse
from sqlparse.tokens import *

try:
    from inspect import isgeneratorfunction
except ImportError:
    # Python < 2.6
    def isgeneratorfunction(obj):
        return bool(
            (inspect.isfunction(object) or inspect.ismethod(object))
            and obj.__code__.co_flags & CO_GENERATOR
        )


try:
    from collections import OrderedDict
except ImportError:
    # Python 2.6
    from lib.collections26 import OrderedDict

break_inversion_subquery = re.compile(
    r"(?i)\s*((?:(?:(?:'[^']*?'|\w+:[^\s]+)\s*)*))" r"(of\s|from\s|)?(.*?)\s*$",
    re.DOTALL | re.UNICODE,
)


find_parenthesis = re.compile(r"""\s*\((.*)\)\s*$""", re.DOTALL | re.UNICODE)
viewdetector = re.compile(
    r"(?i)\s*create\s+(?:temp|temporary)\s+view\s+", re.DOTALL | re.UNICODE
)
inlineop = re.compile(
    r"\s*/\*\*+[\s\n]*((?:def\s+|class\s+).+)[^*]\*\*+/", re.DOTALL | re.UNICODE
)

_statement_cache = OrderedDict()
_statement_cache_size = 1000

inversion = False
# delete reserved SQL keywords that collide with our vtables
if __name__ != "__main__":
    for i in ["EXECUTE", "NAMES", "CACHE", "EXEC", "OUTPUT"]:
        if i in sqlparse.keywords.KEYWORDS:
            del sqlparse.keywords.KEYWORDS[i]


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
        return stack[-1]  # The index of the first unbalanced opening parenthesis
    return -1  # If no unbalanced closing parenthesis found


def flatten_projection(pattern, sql_query):
    while True:
        match = re.search(pattern, sql_query)
        if match and match.groups(0)[0] != "(":
            start_index = match.end()
            end_index = match.end()
            unbalanced_index = find_unbalanced_closing_parenthesis(
                sql_query[start_index:]
            )
            pattern1 = "\s*as\s+\w+"
            match1 = re.search(
                r"^" + pattern1,
                sql_query[match.end() + unbalanced_index + 1 :],
                re.IGNORECASE,
            )
            if match1:
                subq = re.sub(
                    "^" + pattern1,
                    " ",
                    sql_query[match.end() + unbalanced_index + 1 :],
                    re.IGNORECASE,
                )
                sql_query = (
                    sql_query[: match.start()]
                    + sql_query[match.end() : match.end() + unbalanced_index]
                    + subq
                )
            else:
                sql_query = (
                    sql_query[: match.start()]
                    + sql_query[match.end() : match.end() + unbalanced_index]
                    + sql_query[match.end() + unbalanced_index + 1 :]
                )
        else:
            break
    return sql_query


def remove_outer_parentheses(s):
    count = 0
    start = 0
    end = len(s) - 1

    while start <= end and s[start] == "(" and s[end] == ")":
        count += 1
        start += 1
        end -= 1

    if count > 0 and start > end:
        return s[start : end + 1]
    else:
        # No outer parentheses to remove
        return s


def extractprojectstring(sql_query):
    select_index = sql_query.upper().find("SELECT")

    if select_index != -1:
        columns_text = sql_query[select_index + len("SELECT") :].strip()

        # Keep track of opened and closed parentheses
        paren_count = 0
        inside_quotes = False

        # Iterate through the characters
        for i, char in enumerate(columns_text):
            if char == "(" and not inside_quotes:
                paren_count += 1
            elif char == ")" and not inside_quotes:
                paren_count -= 1
            elif char in ('"', "'"):
                inside_quotes = not inside_quotes

            # Check for the end of SELECT statement
            if (
                columns_text.upper().startswith("FROM", i)
                and paren_count == 0
                and not inside_quotes
            ):
                columns_text = columns_text[:i].strip(",")
                break

        return [columns_text]
    else:
        return []


def extractcolumnnames(sql_query):
    select_index = sql_query.upper().find("SELECT")

    if select_index != -1:
        columns_text = sql_query[select_index + len("SELECT") :].strip()

        # Keep track of opened and closed parentheses
        paren_count = 0
        inside_single_quotes = False
        inside_double_quotes = False
        column_names = []

        # Iterate through the characters
        current_column = ""
        for i, char in enumerate(columns_text):
            if char == "(" and not inside_single_quotes and not inside_double_quotes:
                paren_count += 1
                current_column += char
            elif char == ")" and not inside_single_quotes and not inside_double_quotes:
                paren_count -= 1
                current_column += char
            elif char == "'" and not inside_double_quotes:
                inside_single_quotes = not inside_single_quotes
                current_column += char
            elif char == '"' and not inside_single_quotes:
                inside_double_quotes = not inside_double_quotes
                current_column += char
            elif (
                char == ","
                and paren_count == 0
                and not inside_single_quotes
                and not inside_double_quotes
            ):
                column_names.append(current_column.strip())
                current_column = ""
            else:
                current_column += char

            # Check for the end of SELECT statement
            if (
                columns_text.upper().startswith("FROM", i)
                and paren_count == 0
                and not inside_single_quotes
                and not inside_double_quotes
            ):
                if current_column.strip():
                    column_names.append(current_column[:-1].strip())
                    current_column = ""
                break
        if current_column != "":
            column_names.append(current_column.strip())
        return column_names
    else:
        return []


def extract_columns(sql_query, select_range):
    columns = extractcolumnnames(sql_query)
    return [col.strip() for col in columns]


def count_columns_in_udf(udfname, udfstatement):
    pattern = r"" + udfname + "\((.*?)\)"
    columns_match = re.search(pattern, udfstatement, re.DOTALL | re.IGNORECASE)
    if columns_match:
        columns_text = columns_match.group(1).strip()
        coltext = "SELECT " + columns_text + " FROM"
        return len(extractcolumnnames(coltext))
        # Split the columns using commas while excluding commas inside parentheses
        columns = re.split(r",(?![^()]*\))", columns_text)
        return len([col.strip() for col in columns])
    else:
        return 0


def extract_column_names(sql_query, select_range):
    columns = extract_columns(sql_query, select_range)
    pattern = r"\bAS\b"
    patternas = r"\bAS\s+([A-Za-z_][A-Za-z0-9_$]*)\b(?!\s*FROM\b)"
    valid = r"^[A-Za-z_][A-Za-z0-9_$]*$"
    validfunction = r"^[A-Za-z_][A-Za-z0-9_$]*\s*\("
    funcname = r"^([A-Za-z_][A-Za-z0-9_$]*)\s*\("
    # return bool(re.match(pattern, column_name))
    if columns:
        column_names = columns
        cc = 1
        for i, x in enumerate(column_names):
            if not bool(re.search(pattern, x, re.IGNORECASE)):
                if bool(re.match(valid, x)):
                    column_names[i] = x
                else:
                    if re.search(validfunction, x, re.IGNORECASE):  ### perhaps bug
                        column_names[i] = re.match(funcname, x).groups(0)[0]
                    else:
                        column_names[i] = "v" + str(cc)
                        cc += 1
            else:
                match = re.search(patternas, x, re.IGNORECASE)
                if match:
                    coln = match.group(1)
                    if re.match(valid, coln):
                        column_names[i] = coln
                    else:
                        column_names[i] = "v" + str(cc)
                        cc += 1
                else:
                    column_names[i] = "v" + str(cc)
                    cc += 1
        # column_names = [x if not bool(re.search(pattern, x, re.IGNORECASE)) else re.search(patternas, x, re.IGNORECASE).group(1) for x in column_names]
        # column_names = [x if bool(re.match(valid,x)) else 'v'+str(i) for i,x in enumerate(column_names)]
        return column_names
    else:
        return None


# Parse comments for inline ops
def opcomments(s):
    if r"/**" not in str(s):
        return []

    out = []
    constr_comm = None
    for i in s.tokens:
        ui = str(i)
        if type(i) == sqlparse.sql.Comment:
            op = inlineop.match(ui)
            if op != None:
                out.append(op.groups()[0])

        # Construct comment to work around sqlparse bug
        if constr_comm is not None:
            constr_comm += ui

        if type(i) == sqlparse.sql.Token:
            if ui == "/*":
                if constr_comm is not None:
                    constr_comm = None
                else:
                    constr_comm = "/*"
            elif ui == "*/":
                if constr_comm is not None:
                    op = inlineop.match(constr_comm)
                    if op != None:
                        out.append(op.groups()[0])
                    constr_comm = None
    return out


# Top level transform (runs once)
def transform(
    con,
    query,
    multiset_functions=None,
    vtables=[],
    row_functions=[],
    substitute=lambda x: x,
):
    # global inversion REMOVED
    # inversion = False REMOVED
    if type(query) not in (str, str):
        return (query, [], [])
    s = query
    subsquery = substitute(s)

    # Check cache
    if subsquery in _statement_cache:
        # inversion = _statement_cache[subsquery][1] REMOVED
        # return _statement_cache[subsquery][0] REMOVED
        return _statement_cache[subsquery]
    # s = subsquery REMOVED
    enableInlineops = False
    if r"/**" in s:
        enableInlineops = True
    out_vtables = []
    # st = sqlparse.parse(s)
    start_time = time.time()
    st = sqlparse.parse(subsquery)
    end_time = time.time()
    elapsed_time = end_time - start_time
    # print(f"Elapsed sqlparse.parse time: {elapsed_time:.6f} seconds")
    # start_time = time.time()
    trans = Transclass(multiset_functions, vtables, row_functions)
    s_out = ""
    sqp = ("", [], [])
    inlineops = []
    # start_time = time.time()
    for s in st:
        # delete question mark
        strs = re.match(r"(.*?);*\s*$", str(s), re.DOTALL | re.UNICODE).groups()[0]
        st1 = sqlparse.parse(strs)
        if len(st1) > 0:
            if enableInlineops:
                inlineops.append(opcomments(st1[0]))
            sqp = trans.rectransform(con, st1[0])
            strs = str(sqp[0])
            s_out += strs
            s_out += ";"

            # Detect create temp view and mark its vtables as permanent
            if viewdetector.match(strs):
                out_vtables += [x + (False,) for x in sqp[1]]
            else:
                out_vtables += sqp[1]

    result = (s_out, vt_distinct(out_vtables), sqp[2], inlineops)
    if len(_statement_cache) < _statement_cache_size:
        _statement_cache[subsquery] = result  # (result, inversion) REMOVED
    else:
        _statement_cache.popitem(last=False)
        _statement_cache[subsquery] = result  # (result,inversion)
    #    end_time = time.time()
    #    elapsed_time = end_time - start_time

    #    print(f"Elapsed full transform time: {elapsed_time:.6f} seconds")
    return result


class Transclass:
    direct_exec = []
    multiset_functions = None
    vtables = []
    row_functions = []

    def __init__(self, multiset_functions=None, vtables=[], row_functions=[]):
        self.direct_exec = []
        self.multiset_functions = multiset_functions
        self.vtables = vtables
        self.row_functions = row_functions

    # recursive transform
    def rectransform(self, con, s, s_orig=None):
        # global inversion REMOVED
        # print('lala: ',s)
        start_time = time.time()
        if not (
            re.search(
                r"(?i)(select|"
                + "|".join([x for x in self.vtables])
                + "|"
                + "|".join(self.multiset_functions.keys())
                + "|"
                + "|".join(self.row_functions)
                + ")",
                str(s),
                re.UNICODE,
            )
        ):
            return str(s), [], self.direct_exec
        out_vtables = []
        if s_orig is None:
            s_orig = s

        query = None
        # Expand functions with spaces between them and their parenthesis
        for t in s_orig.tokens:
            tfm = re.match("(\w+)\s\(", str(t), re.UNICODE)
            if (
                isinstance(t, sqlparse.sql.Function)
                and tfm
                and (
                    tfm.groups()[0] in self.vtables
                    or tfm.groups()[0] in self.row_functions
                )
            ):
                tidx = s_orig.token_index(t)
                s_orig.tokens[tidx : tidx + 1] = t.tokens

        fs = [x for x in expand_tokens(s)]
        # Process external_query VTs
        tmatch = re.match(r"\s*(\w+)\s+(.*|$)", str(s), re.DOTALL | re.UNICODE)
        if tmatch is not None and tmatch.groups()[0].lower() in self.vtables:
            op_for_inv = tmatch.groups()[0].lower()
            if hasattr(self.vtables[op_for_inv], "external_query"):
                rest = tmatch.groups()[1]
                op_for_inv = str(op_for_inv)
                params, preposition, subq = break_inversion_subquery.match(
                    rest
                ).groups()
                if subq != "":
                    paramslist = [format_query(subq)]
                else:
                    paramslist = []
                paramslist += [
                    format_param("".join(x))
                    for x in re.findall(r"'([^']*?)'|(\w+:[^\s]+)", params, re.UNICODE)
                ]
                inv_s = ",".join(paramslist)
                vname = vt_name(op_for_inv)
                self.direct_exec += [(op_for_inv, paramslist, subq)]
                s_orig.tokens[
                    s_orig.token_index(s.tokens[0]) : s_orig.token_index(s.tokens[-1])
                    + 1
                ] = [sqlparse.sql.Token(Token.Keyword, "select * from " + vname + " ")]
                return (
                    str(s),
                    vt_distinct([(vname, op_for_inv, inv_s)]),
                    self.direct_exec,
                )

        # Process internal parenthesis
        for t in fs:
            if type(t) is sqlparse.sql.Parenthesis:
                subq = find_parenthesis.match(str(t))
                if subq != None:
                    subq = subq.groups()[0]
                    t.tokens = sqlparse.parse(subq)[0].tokens
                    out_vtables += self.rectransform(con, t)[1]
                    t.tokens[0:0] = [sqlparse.sql.Token(Token.Punctuation, "(")]
                    t.tokens.append(sqlparse.sql.Token(Token.Punctuation, ")"))

        # genfuncdatatypes = [] REMOVED
        # ss = str(s) REMOVED
        t = re.match(r"\s*(\w+)(\s+.*|$)", str(s), re.DOTALL | re.UNICODE)
        tmatch2 = re.match(
            r"cast\s*\(\*(\w+)\s+(.*|$)", str(s), re.DOTALL | re.UNICODE | re.IGNORECASE
        )
        if (t != None or tmatch2 != None) and t.groups()[
            0
        ].lower() in self.row_functions:
            # inversion = True REMOVED
            op_for_inv = t.groups()[0]
            rest = t.groups()[1]
            params, preposition, subq = break_inversion_subquery.match(rest).groups()
            paramslist = [
                format_param("".join(x))
                for x in re.findall(r"'([^']*?)'|(\w+:[^\s]+)", params, re.UNICODE)
            ]
            if subq != "":
                if len(preposition) > 0:
                    subq, v, dv = self.rectransform(con, sqlparse.parse(subq)[0])
                    out_vtables += v
                    paramslist += ["(" + subq + ")"]
                else:
                    paramslist += [format_param(subq)]
            inv_s = "SELECT " + op_for_inv + "(" + ",".join(paramslist) + ")"
            subs = sqlparse.parse(inv_s)[0]
            s_orig.tokens[
                s_orig.token_index(s.tokens[0]) : s_orig.token_index(s.tokens[-1]) + 1
            ] = subs.tokens
            s = subs
        fs = [x for x in expand_tokens(s)]

        # Process vtable inversion
        for t in fs:
            if t.ttype == Token.Keyword.DML:
                break
            start_time = time.time()
            strt = str(t).lower()
            if strt in self.vtables:
                # print ("FOUND INVERSION:", strt, fs)
                # inversion = True REMOVED
                tindex = fs.index(t)
                # Break if '.' exists before vtable
                if tindex > 0 and str(fs[tindex - 1]) == ".":
                    break
                op_for_inv = strt
                try:
                    rest = "".join([str(x) for x in fs[tindex + 1 :]])
                except KeyboardInterrupt:
                    raise
                except:
                    rest = ""
                params, preposition, subq = break_inversion_subquery.match(
                    rest
                ).groups()
                orig_subq = subq
                if subq != "":
                    subq, v, dv = self.rectransform(con, sqlparse.parse(subq)[0])
                    out_vtables += v
                end_time = time.time()
                if not hasattr(self.vtables[strt], "external_stream"):
                    # if subq != '':   #REMOVED perhaps a bug with table udfs without a query
                    #    paramslist = [format_query(subq)]
                    # else:
                    #    paramslist = []
                    # start_time = time.time()
                    paramslist = []
                    scalarparamslist = [
                        format_param("".join(x))
                        for x in re.findall(
                            r"'([^']*?)'|(\w+:[^\s]+)", params, re.UNICODE
                        )
                    ]
                    # paramslist += scalarparamslist
                    queryparam = format_query(
                        subq
                    )  # REMOVED perhaps this is a bug to remove

                    if scalarparamslist != []:
                        if queryparam == "()":
                            args_str = ", ".join(scalarparamslist)
                            inv_s = (
                                "".join([str(x) for x in fs[: fs.index(t)]])
                                + "SELECT * FROM "
                                + op_for_inv
                                + "("
                                + args_str
                                + ")"
                            )
                        else:
                            params = {}
                            params["'scpar'"] = scalarparamslist
                            lala = json.dumps(params)
                            single_quoted_string = "'" + lala.replace('"', "'") + "'"
                            mtype = ""
                            datatypes = []
                            dtypes = op_for_inv[
                                len(self.vtables[strt].__name__) + 1 :
                            ].split("_")
                            if dtypes == [""]:
                                try:
                                    dtypes = functions.stfunctions["vtable"][
                                        self.vtables[strt].__name__
                                    ].datatype
                                except:
                                    raise ValueError(
                                        "YeSQLError: No return datatype defined for function "
                                        + self.vtables[strt].__name__
                                    )
                            for num, tt in enumerate(dtypes):
                                if tt.lower() == "string" or tt.lower == "char*":
                                    mtype += (
                                        self.vtables[strt].__name__
                                        + "_"
                                        + str(num)
                                        + "  STRING,"
                                    )
                                    datatypes.append(datatypemap["STRING"])
                                elif tt.lower() == "int" or tt.lower() == "int*":
                                    mtype += (
                                        self.vtables[strt].__name__
                                        + "_"
                                        + str(num)
                                        + "  INT,"
                                    )
                                    datatypes.append(datatypemap["INT"])
                                elif (
                                    tt.lower() == "bigint" or tt.lower() == "long long*"
                                ):
                                    mtype += (
                                        self.vtables[strt].__name__
                                        + "_"
                                        + str(num)
                                        + "  BIGINT,"
                                    )
                                    datatypes.append(datatypemap["BIGINT"])
                                elif tt.lower() == "float" or tt.lower() == "double*":
                                    mtype += (
                                        self.vtables[strt].__name__
                                        + "_"
                                        + str(num)
                                        + "  float,"
                                    )
                                    datatypes.append(datatypemap["FLOAT"])
                            mtype = mtype.strip(",")
                            op_for_inv = (
                                "udf"
                                + hashlib.md5(
                                    strt.encode() + single_quoted_string.encode()
                                ).hexdigest()
                            )
                            newtablename = functions.register_table(
                                op_for_inv,
                                self.vtables[strt].__name__,
                                self.vtables[strt],
                                mtype,
                                datatypes,
                                con,
                                functions.Connection,
                                False,
                                single_quoted_string,
                            )
                            ##TODO edit register_table so that it passes params in single_quoted_string to the udf wrapper, rename also table UDF using a name including these params
                            # queryparam = format_query(subq)
                            # queryparam = f'(SELECT {single_quoted_string}, * FROM {format_query(subq)} Χ) '
                            inv_s = (
                                "".join([str(x) for x in fs[: fs.index(t)]])
                                + "SELECT * FROM "
                                + op_for_inv
                                + "("
                                + queryparam
                                + ")"
                            )
                    else:
                        if queryparam != "()":
                            inv_s = (
                                "".join([str(x) for x in fs[: fs.index(t)]])
                                + "SELECT * FROM "
                                + op_for_inv
                                + "("
                                + queryparam
                                + ")"
                            )
                        else:
                            inv_s = (
                                "".join([str(x) for x in fs[: fs.index(t)]])
                                + "SELECT * FROM "
                                + op_for_inv
                                + "()"
                            )
                else:
                    paramslist = [
                        format_param("".join(x))
                        for x in re.findall(
                            r"'([^']*?)'|(\w+:[^\s]+)", params, re.UNICODE
                        )
                    ]
                    inv_s = (
                        "".join([str(x) for x in fs[: fs.index(t)]])
                        + "SELECT * FROM "
                        + op_for_inv
                        + "("
                        + ",".join(paramslist)
                        + ") "
                        + subq
                    )
                subs = sqlparse.parse(inv_s)[0]
                self.direct_exec += [(op_for_inv, paramslist, orig_subq)]
                s_orig.tokens[
                    s_orig.token_index(s.tokens[0]) : s_orig.token_index(s.tokens[-1])
                    + 1
                ] = subs.tokens
                s = subs
                # end_time = time.time()
                elapsed_time = end_time - start_time
                # print(f"Elapsed my rectransform vtable inversion time: {elapsed_time:.6f} seconds")
                break
        # find first select
        s_start = s.token_next_match(0, Token.Keyword.DML, r"(?i)select", True)
        if s_start is not None:
            # find keyword that ends substatement
            s_end = s.token_next_match(
                s.token_index(s_start),
                Token.Keyword,
                (
                    r"(?i)union",
                    r"(?i)order",
                    r"(?i)limit",
                    r"(?i)intersect",
                    r"(?i)except",
                    r"(?i)having",
                ),
                True,
            )
            if len(s.tokens) < 3:
                return str(s), vt_distinct(out_vtables), self.direct_exec
            if s_end is None:
                if s.tokens[-1].value == ")":
                    s_end = s.tokens[-2]
                else:
                    s_end = s.tokens[-1]
            else:
                if s.token_index(s_end) + 1 >= len(s.tokens):
                    raise functions.YeSQLError(
                        "'" + str(s_end).upper() + "' should be followed by something"
                    )
                out_vtables += self.rectransform(
                    con,
                    sqlparse.sql.Statement(
                        s.tokens_between(
                            s.tokens[s.token_index(s_end) + 1], s.tokens[-1]
                        )
                    ),
                    s,
                )[1]
                s_end = s.tokens[s.token_index(s_end) - 1]
            query = sqlparse.sql.Statement(s.tokens_between(s_start, s_end))
        else:
            return str(s), vt_distinct(out_vtables), self.direct_exec
        # find from and select_parameters range
        from_range = None
        from_start = query.token_next_match(0, Token.Keyword, r"(?i)from", True)
        # process virtual tables in from range
        if from_start is not None and 1 != 1:
            from_end = query.token_next_by_instance(
                query.token_index(from_start), sqlparse.sql.Where
            )
            if from_start == query.tokens[-1]:
                raise functions.YeSQLError(
                    "Error in FROM range of: '" + str(query) + "'"
                )
            if from_end is None:
                from_end = query.tokens[-1]
                from_range = sqlparse.sql.Statement(
                    query.tokens_between(
                        query.tokens[query.token_index(from_start) + 1], from_end
                    )
                )
            else:
                from_range = sqlparse.sql.Statement(
                    query.tokens_between(
                        query.tokens[query.token_index(from_start) + 1],
                        from_end,
                        exclude_end=True,
                    )
                )
            for t in [
                x
                for x in expand_type(
                    from_range, (sqlparse.sql.Identifier, sqlparse.sql.IdentifierList)
                )
            ]:
                if str(t).lower() in ("group", "order"):
                    break
                if type(t) is sqlparse.sql.Function:
                    vname = vt_name(str(t))
                    fname = t.tokens[0].get_real_name().lower()
                    if fname in self.vtables:
                        out_vtables += [(vname, fname, str(t.tokens[1])[1:-1])]
                        t.tokens = [sqlparse.sql.Token(Token.Keyword, vname)]
                    else:
                        raise functions.YeSQLError(
                            "Virtual table '" + fname + "' does not exist"
                        )

        if from_start is not None:
            select_range = sqlparse.sql.Statement(
                query.tokens_between(query.tokens[1], from_start, exclude_end=True)
            )
        else:
            select_range = sqlparse.sql.Statement(
                query.tokens_between(query.tokens[1], query.tokens[-1])
            )
        # expandfuncs = []
        # Process EXPAND functions and CASTS
        # multisetindex = []
        # expanddatatypes = []
        genfuncdatatypes = []
        ss = str(query)
        multisetfuncs = -1
        for t in flatten_with_type(select_range, sqlparse.sql.Function):
            if hasattr(t.tokens[0], "get_real_name"):
                funcname = t.tokens[0].get_real_name()
            else:
                funcname = str(t.tokens[0])
            funcname = funcname.lower().strip()
            res = re.search(
                rf"cast\s*\(\s*({funcname})\s*\(",
                ss,
                flags=re.VERBOSE | re.UNICODE | re.IGNORECASE,
            )

            if res is not None:
                funcdata = res.groups()
                result_index = find_unbalanced_closing_parenthesis(ss[res.end() :])
                datatype = re.search(
                    r"\s*as\s*((?:\s*[a-zA-Z0-9, _\s]+\s*)|(?:\([a-zA-Z0-9, _\s]+)\s*\))\s*\)",
                    ss[result_index + res.end() :],
                    flags=re.VERBOSE | re.UNICODE | re.IGNORECASE,
                )
                funcdata += datatype.groups()
                if (
                    funcdata[0] in functions.functions["row"]
                    and isgeneratorfunction(functions.functions["row"][funcdata[0]])
                ) or (
                    funcdata[0] in functions.functions["aggregate"]
                    and isgeneratorfunction(
                        functions.functions["aggregate"][funcdata[0]].final
                    )
                ):
                    genfuncdatatypes.append(datatype.groups())
                    ss = (
                        ss[: result_index + res.end() + datatype.start()]
                        + ss[result_index + res.end() + datatype.end() :]
                    )
                    ss = re.sub(
                        "cast\s*\(",
                        "",
                        ss,
                        count=1,
                        flags=re.VERBOSE | re.UNICODE | re.IGNORECASE,
                    )

            if (
                funcname in self.multiset_functions
                and udftypes[functions.functiontypes[funcname]] == 2
            ):
                multisetfuncs += 1
                # expandfuncs.append(fname)
                t = s_orig.group_tokens(
                    sqlparse.sql.Parenthesis, s_orig.tokens_between(s_start, s_end)
                )
                vname = vt_name(str(t))
                column_names = extract_column_names(ss, select_range)
                column_full_names = extract_columns(ss, select_range)
                # out_vtables += [(vname, 'expand_'+ '_'.join([x for x in expanddatatypes]), format_query(t))]
                lines, fname_returnx, fname, fcalls = parseplan(
                    ss, con, self.multiset_functions
                )
                expanddatatypes = []
                multisetindex = []
                num = 0
                for expandtypes in lines[0]:
                    if expandtypes[0] not in fname_returnx:
                        expanddatatypes.append(plandatatypes[expandtypes[1]])
                        num += 1
                    else:
                        index = fname_returnx.index(expandtypes[0])
                        rets = None
                        try:
                            rets = [
                                part.strip()
                                for part in genfuncdatatypes[multisetfuncs][0].split(
                                    ","
                                )
                            ]
                            del genfuncdatatypes[multisetfuncs]
                            multisetindex.append(num)
                            multisetindex.append(len(rets))
                            num += len(rets)
                            for ret in rets:
                                if ret.lower() == "string":
                                    expanddatatypes.append("STRING")
                                elif ret.lower() == "int":
                                    expanddatatypes.append("INT")
                                elif ret.lower() == "bigint":
                                    expanddatatypes.append("BIGINT")
                                elif (
                                    ret.lower() == "float"
                                    or ret == "real"
                                    or ret == "double"
                                ):
                                    expanddatatypes.append("FLOAT")
                            try:
                                functions.functions["row"][
                                    fname[index]
                                ].returntype = None
                            except:
                                functions.functions["aggregate"][
                                    fname[index]
                                ].returntype = None
                        except:
                            try:
                                rets = self.multiset_functions[fname[index]].__args__
                                i = len(rets)
                                while i > 0 and rets[i - 1] is type(None):
                                    i -= 1
                                rets = rets[:i]
                                multisetindex.append(num)
                                multisetindex.append(len(rets))
                                num += len(rets)
                                for ret in rets:
                                    if ret is str:
                                        expanddatatypes.append("STRING")
                                    elif ret is int:
                                        expanddatatypes.append("INT")
                                    elif isinstance(ret, types.FunctionType):
                                        expanddatatypes.append("BIGINT")
                                    elif ret is float:
                                        expanddatatypes.append("FLOAT")
                            except:
                                expanddatatypes.append(
                                    "STRING"
                                )  # no provided type, use string datatype
                                multisetindex.append(num)
                                multisetindex.append(1)
                                num += 1
                        colnames = []
                        yy = 0
                        for i, x in enumerate(column_names):
                            if yy < len(multisetindex) and i == multisetindex[yy]:
                                yy += 1
                                for k in range(multisetindex[yy]):
                                    colnames.append(
                                        "v"
                                        + str(i + k + 1)
                                        + " as "
                                        + x
                                        + "_"
                                        + str(k + 1)
                                    )
                                yy += 1
                            elif i > multisetindex[yy - 2]:
                                colnames.append(
                                    "v"
                                    + str(
                                        int(
                                            i
                                            + 1
                                            + sum(multisetindex[1::2])
                                            - len(multisetindex) / 2
                                        )
                                    )
                                    + " as "
                                    + x
                                )
                            else:
                                colnames.append("v" + str(i + 1) + " as " + x)
                udfargs = []
                i = 0
                for f in fname:
                    while i < len(column_full_names):
                        if f in column_full_names[i]:
                            udfargs.append(
                                count_columns_in_udf(f, column_full_names[i])
                            )
                            break
                        i += 1
                query = format_query(ss)
                funames = []
                aggr = False
                for ff in fname:
                    if ff not in list(functions.functions["aggregate"].keys()):
                        funames.append(ff)
                    else:
                        aggr = True
                pattern1 = r"(\b(" + "|".join(funames) + ")\s*\()"
                query = flatten_projection(pattern1, query)
                paramscount = 1
                if query == "(select )":
                    query = ""
                    paramscount = 0
                # vname = '(select '+",".join([x for x in colnames])+' from expand_'+ '_'.join([x for x in expanddatatypes])+'('+ format_query(t)+')) ' + vname
                if paramscount == 0:
                    udfargs = []
                # print('vname: ', vname)
                vname = (
                    "(select "
                    + ",".join([x for x in colnames])
                    + " from expand_"
                    + fname[0]
                    + "_".join([x for x in expanddatatypes])
                    + "".join([str(x) for x in udfargs])
                    + "("
                    + query
                    + ")) "
                    + vname
                )
                # print('vname2: ', vname)
                ss = (
                    "select "
                    + ",".join([x for x in colnames])
                    + " from expand_"
                    + fname[0]
                    + "_".join([x for x in expanddatatypes])
                    + "("
                    + query
                    + ") "
                    + vname
                )
                out_vtables += [
                    (
                        vname,
                        "expand_" + fname[0] + "_".join([x for x in expanddatatypes]),
                        format_query(ss),
                    )
                ]
                if aggr:
                    register_expand(
                        con,
                        expanddatatypes,
                        multisetindex,
                        vname,
                        fcalls,
                        functions.functions["aggregate"][fcalls[0].split(".")[-1]],
                        udfargs,
                        paramscount,
                        aggr,
                    )
                else:
                    register_expand(
                        con,
                        expanddatatypes,
                        multisetindex,
                        vname,
                        fcalls,
                        self.multiset_functions[fcalls[0].split(".")[-1]],
                        udfargs,
                        paramscount,
                        aggr,
                    )
                # con.executetrace('create temp table ' +vname+ ' as select * from expand_'+ '_'.join([x for x in expanddatatypes])+'('+ format_query(t)+') on commit preserve rows;')
                #    if fname in str(nextline) and 'batcapi' in str(nextline):
                #        print(nextline)
                # print ('select * from expand('+format_query(t)+')')
                temp = str(s)
                s_orig.tokens[s_orig.token_index(t)] = sqlparse.sql.Token(
                    Token.Keyword, "select * from " + vname + " "
                )
                break
                # return vname2, vt_distinct(out_vtables), self.direct_exec
            else:
                pass
                # s_orig.tokens[s_orig.token_index(t)] = sqlparse.sql.Token(Token.Keyword, str(ss))
                # print('kkkkkkkkkkkkk')
        # print('jakfref: ', vname2, str(s))
        if multisetfuncs != -1 and str(s) == temp:
            # print('lala')
            s = self.rectransform(con, s)[0]
        # if multisetfuncs == -1:
        #    return str(ss), vt_distinct(out_vtables), self.direct_exec
        end_time = time.time()
        elapsed_time = end_time - start_time
        # print(f"Elapsed my rectransform time: {elapsed_time:.6f} seconds")
        return str(s), vt_distinct(out_vtables), self.direct_exec


def vt_name(s):
    tmp = re.sub(
        r"([^\w])", "_", "vt_" + str(zlib.crc32(s.encode("utf-8"))), re.UNICODE
    )
    return re.sub(r"_+", "_", tmp, re.UNICODE)


def format_query(s):
    q = "(" + str(s).replace("'", "'") + ")"
    q = q.replace("\n", " ")
    return q


def format_param_scalar(s):
    return str(s).replace("'", "''")


def format_param(s):
    return "'" + str(s).replace("'", "''") + "'"


def format_identifiers(s):
    return str(s).replace(" ", "").replace("\t", "")


def flatten_with_type(inpt, clss):
    """Generator yielding ungrouped tokens.

    This method is recursively called for all child tokens.
    """
    for token in inpt.tokens:
        if isinstance(token, clss):
            yield token
        else:
            if token.is_group() or type(token) is sqlparse.sql.Parenthesis:
                for i in flatten_with_type(token, clss):
                    yield i


def expand_type(inpt, clss):
    """Generator yielding ungrouped tokens.

    This method is recursively called for all child tokens.
    """
    for token in inpt.tokens:
        if token.is_group() and isinstance(token, clss):
            for i in expand_type(token, clss):
                yield i
        else:
            yield token


def expand_tokens(inpt):
    """Generator yielding ungrouped tokens recursively for all token groups."""
    for token in inpt.tokens:
        if token.is_group() and isinstance(
            token,
            (
                sqlparse.sql.Identifier,
                sqlparse.sql.IdentifierList,
                sqlparse.sql.Where,
                sqlparse.sql.Function,
            ),
        ):
            for i in expand_tokens(token):
                yield i
        else:
            yield token


def vt_distinct(vt):
    vtout = OrderedDict()
    for i in vt:
        if i[0] not in vtout:
            vtout[i[0]] = i
        else:
            if not vtout[i[0]][-1] == False:
                vtout[i[0]] = i

    return list(vtout.values())


if __name__ == "__main__":

    sql = []
    multiset_functions = ["nnfunc1", "nnfunc2", "apriori", "ontop", "strsplit"]

    def file():
        pass

    file.external_stream = True

    def execv():
        pass

    execv.no_results = True

    vtables = {
        "file": file,
        "lnf": True,
        "funlalakis": True,
        "filela": True,
        "sendto": True,
        "helpvt": True,
        "output": True,
        "names": True,
        "cache": True,
        "testvt": True,
        "exec": execv,
        "flow": True,
    }
    row_functions = [
        "help",
        "set",
        "execute",
        "var",
        "toggle",
        "strsplit",
        "min",
        "ifthenelse",
        "keywords",
    ]

    sql += ["select a,b,(apriori(a,b,c,'fala:a')) from lalatable"]
    sql += [
        "create table a from select a,b,(apriori(a,b,c,'fala:a')) from lalatable, lala14, lala15"
    ]
    sql += [
        "create table a from select a,b,(apriori(a,b,c,'fala:a')) from lalatable, lala14, lala15"
    ]
    sql += [
        "select a,b,(apriori(a,b,c,'fala:a')) from lalatable where a=15 and b=23 and c=(1234)"
    ]
    sql += [
        "select a,b,(apriori(a,b,c,'fala:a')) from lalatable where a=15 and b=23 and c=(1234) group by a order by"
    ]
    sql += [
        "select a,b,(apriori(a,b,c,'fala:a')) from ('asdfadsf') where a=15 and b=23 and c=(1234) group by a order by"
    ]
    sql += [
        "select a,b,(apriori(a,b,c,'fala:a')) from ('asdfadsf') where a=15 and b=23 and c=(1234) group by a order by b union select a,b from funlalakis('1234'), (select a from lnf('1234') )"
    ]
    sql += [
        "select c1,c2 from file('test.tsv', 'param1');select a from filela('test.tsv') group by la"
    ]
    sql += ["insert into la values(1,2,3,4)"]
    sql += ["select apriori(a) from (select apriori('b') from table2)"]
    sql += [
        "select userid, top1, top2 from (select userid,ontop(3,preference,collid,preference) from colpreferences group by userid)order by top2 ; "
    ]
    sql += ["select ontop(a), apriori(b) from lala"]
    sql += ["select ontop(a) from (select apriori(b) from table) order by a"]
    sql += [
        "select userid,ontop(3,preference,collid,preference),ontop(1,preference,collid) from colpreferences group by userid;"
    ]
    sql += ["create table lala as select apriori(a) from table;"]
    sql += [
        "create table lila as select userid,ontop(3,preference,collid,preference),ontop(1,preference,collid) from colpreferences group by userid; "
    ]
    sql += ["select * from file(test.txt)"]
    sql += ["select sum(b) from test_table group by a pivot b,c"]
    sql += ["select * from (helpvt lala)"]
    sql += ["output 'list'"]
    sql += ["(help lala)"]
    sql += [r"select * from tab1 union help 'lala'"]
    sql += [r"select * from file('list'),(select * from file('list'))"]
    sql += [r"create table ta as help list"]
    sql += [r"select * from (help lala)"]
    sql += [r"output 'lala' select apriori(a,b) from extable"]
    sql += [r"select apriori(a,b) from extable"]
    sql += [r"select * from file('/lala','param1:t')"]
    sql += [r"output '/lala' 'param1' select * from tab"]
    sql += [r"select apriori(a,b) from file(/lala/lalakis)"]
    sql += [
        "(select a from (sendto 'fileout.lala' 'tsv' select * from file('file.lala')))"
    ]
    sql += [
        "sendto 'lala1' sendto 'fileout.lala' 'tsv' select * from file('file.lala'))"
    ]
    sql += ["help 'lala'"]
    sql += ["names file 'lala'; helpvt lala"]
    sql += [r"select * from file() as a, file() as b;"]
    sql += [r"select file from (file 'alla) as lala"]
    sql += [r"  .help select * from file('lsls')"]
    sql += [r"  .execute select * from file('lsls')"]
    sql += [r"limit 1"]
    sql += [r"file 'lala'"]
    sql += [r"select * from lala union file 'lala' union file 'lala'"]
    sql += [r"file 'lala' limit 1"]
    sql += [r"create table lala file 'lala'"]
    sql += [r"SELECT * FROM (file 'lala')"]
    sql += [r"(file 'lala') union (file 'lala1')"]
    sql += [r"select (5+5) from (file 'lala1')"]
    sql += [
        r"select * from ( output 'bla' select * from file('collection-general.csv','dialect:line') where rowid!=1 ) "
    ]
    sql += [r"select * from testtable where x not in (file 'lalakis')"]
    # sql+=[r".help ασδαδδ"]
    sql += [r"names (file 'testfile')"]
    # sql+=[r"select * from (select lala from table limit)"]
    sql += [
        r"""create table session_to_country(
	sesid text NOT NULL primary key,
	geoip_ccode text
);    """
    ]
    sql += [
        r"""create table ip_country as select iplong,CC from (cache select cast(C3 as integer) as ipfrom,cast(C4 as
integer) as ipto, C5 as CC from file('file:GeoIPCountryCSV_09_2007.zip','compression:t','dialect:csv') ),tmpdistlong
where iplong>=ipfrom and iplong <=ipto;
"""
    ]
    sql += [r"cache select * from lala;"]
    sql += [r"var 'lala' from var 'lala1'"]
    sql += [r"toggle tracing"]
    sql += [r"select strsplit('8,9','dialect:csv')"]
    sql += [r"testvt"]
    sql += [r"select date('now')"]
    sql += [r"exec select * from lala"]
    sql += [r"var 'usercc' from select min(grade) from (testvt) where grade>5;"]
    sql += [r"var 'usercc' from select 5;"]
    sql += [r"(exec flow file 'lala' 'lala1' asdfasdf:asdfdsaf);"]
    sql += [
        r"UPDATE merged_similarity SET  merged_similarity = ((ifthenelse(colsim,colsim,0)*0.3)+(ifthenelse(colsim,colsim,0)*0.3))"
    ]
    sql += [r"toggle tracing ;"]
    sql += [
        r"select sesid, query from tac group by sesid having keywords('query')='lala'"
    ]
    sql += [
        r"select sesid, query from tac group by sesid having keywords('query')='lala' union select * from file('lala')"
    ]
    sql += [
        r"select * from (select 5 as a) where a=4 or (a=5 and a not in (select 3));"
    ]
    sql += [r"select * from a where ((a.r in (select c1 from f)));"]
    sql += [r"select upper(a.output) from a"]
    sql += [r"select upper(execute) from a"]
    sql += [r"exec select a.5 from (flow file 'lala')"]
    sql += [r"select max( (select 5))"]
    sql += [r"cache select 5; create temp view as cache select 7; cache select 7"]
    sql += [r"select * from /** def lala(x): pass **/ tab"]
    sql += [r"select * from /* def lala(x): pass **/ tab"]
    sql += [r"/** def lala():return 6 **/ \n"]
    sql += [r"/** def lala():return 6 **/ "]

    for s in sql:
        print("====== " + str(s) + " ===========")
        a = transform(s, multiset_functions, vtables, row_functions)
        print("Query In:", s)
        print("Query Out:", a[0].encode("utf-8"))
        print("Vtables:", a[1])
        print("Direct exec:", a[2])
        print("Inline Ops:", a[3])
