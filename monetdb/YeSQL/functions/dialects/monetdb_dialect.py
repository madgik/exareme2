import re
import warnings

import pymonetdb

udfs_cache = {}

udftypes = (
    {  #  keys are udf subtypes, values:  0 for scalar, 1 for aggregate, 2 for table
        1: 0,
        2: 1,
        3: 2,
        4: 2,
        5: 2,
        6: 2,
        7: 2,
        8: 2,
        9: 2,
        10: 2,
    }
)

# the keys are the sql strings used for the datatypes in a create function statement, and the values are the corresponding C datatypes used in the db's C UDF engine
datatypemap = {
    "STRING": "char**",
    "INT": "int*",
    "FLOAT": "double*",
    "BOOL": "bool*",
    "TINYINT": "int8_t *",
    "SMALLINT": "short *",
    "BIGINT": "long long *",
    "HUGEINT": "__int128 *",
}

# keys are the datatypes as they are written in the query plan, for example in mal plan string datatype is written 'str'
plandatatypes = {
    "sht": "SMALLINT",
    "bte": "TINYINT",
    "bit": "BOOL",
    "str": "STRING",
    "lng": "BIGINT",
    "hge": "HUGEINT",
    "dbl": "FLOAT",
    "int": "INT",
}


# basic datatypes that a python UDF returns and how this translate to the database data types
db_sql_string = "STRING"
db_sql_int = "INT"
db_sql_double = "FLOAT"
db_sql_float = "FLOAT"


def connection_string(user, password, host, db, port):
    return db, host, port, user, password


def createConnection(Connection, user, password, host, db, port):
    # print(Connection ,user, password, host, db, port)
    connection = Connection(
        username=user,
        password=password,
        hostname=host,
        database=db,
        port=port,
        autocommit=True,
    )
    connection.connect()
    return connection


# dbConnection = pymonetdb.connect


class dbConnection:
    def __init__(self, username, password, hostname, database, port, autocommit):
        self.username = username
        self.password = password
        self.hostname = hostname
        self.database = database
        self.port = port
        self.autocommit = autocommit
        self.con = None

    def connect(self):
        self.con = pymonetdb.connect(
            username=self.username,
            password=self.password,
            hostname=self.hostname,
            database=self.database,
            port=self.port,
            autocommit=self.autocommit,
        )
        return self.con

    def cursor(self):
        return self.con.cursor()

    def close(self):
        self.con.close()


dbconnect = pymonetdb.connect

## the following 2 functions are used to support specific functionalities

# funcexpand is a no op operation that is used to support functions that are applied to one row at a time or one group at a time and return multiple rows
# In YeSQL we support implementation of such UDFs as scalars/aggregates.
# Since several engines do not support this we use the following dummy function instead, we run the query using this function with explain to get the query plan
# and then we extract from the query plan the datatypes to rewrite the query using table UDF.

funcexpand = """
    CREATE or replace FUNCTION expand(*)
            RETURNS TABLE (i STRING)
            LANGUAGE C {
            i->initialize(i, 1);
            i->data[0] =  "4b41025d848e3c0e54f6dbd07fde1777";
            return 0;
           };
    """

# funcmytable is a UDF used to run a query and keep its result in the UDFs engine global state, it does not return the results of the query but a hash key to a global dict where the results is stored

funcmytable = """
         CREATE or replace FUNCTION mytable(*)
            RETURNS TABLE (i STRING)
            LANGUAGE C {
            #pragma CFLAGS -I$CURRENT/YeSQL/functions
            #include "udfs.h"
            #include <stdio.h>
            #include <stdlib.h>
            #include <dlfcn.h>
            #include <link.h>

            void* py_handle = dlopen("/home/johnfouf/yesql_repo/yesql/YeSQL/functions/libwrappedudfs2.so", RTLD_LAZY | RTLD_GLOBAL | RTLD_DEEPBIND);
            if (!py_handle) {
                fprintf(stderr, "dlopen failed: %s\n", dlerror());
                return NULL;
            }
            int (*tablematerwrapper)(int, ArrayInfo1*, int, ArrayInfo1*, int) = (int (*)(int, ArrayInfo1*, int, ArrayInfo1*, int)) dlsym(py_handle, "tablematerwrapper");

            ArrayInfo arrays[paramsnum];
            ArrayInfo arraysresult[outputsnum];

            for (int i = 0; i < paramsnum; ++i) {
                 arrays[i].type = ((struct cudf_data_struct_str*)__inputs[i])->datatype;
                 arrays[i].array = ((struct cudf_data_struct_str*)__inputs[i])->data;
                 arrays[i].size = ((struct cudf_data_struct_str*)__inputs[i])->count;
                             }
            for (int i = 0; i < outputsnum; ++i) {
                 arraysresult[i].type = ((struct cudf_data_struct_str*)__outputs[i])->datatype;
                          }
            int ret = tablematerwrapper(paramsnum, arrays, outputsnum, arraysresult, arg2.count);
            i->initialize(i, 1);
            i->data[0] =  ((char**)arraysresult[0].array)[0];
            free(arraysresult[0].array);
            free(arrays);
            free(arraysresult);
            return 0;
           };

"""


# get the list of the UDFs defined in the data engine
def getudfnames(ccur, functions):
    ccur.executetrace(
        "select distinct name,type from functions where type = 1 or type = 3 or type = 5;"
    )
    try:
        while True:
            func = ccur.next()
            if func[0] != "insert":
                if func[1] == 1:
                    functions["row"][func[0]] = "sys"
                elif func[1] == 3:
                    functions["aggregate"][func[0]] = "sys"
                elif func[1] == 5:
                    functions["vtable"][func[0]] = "sys"
    except StopIteration:
        pass
    return ccur


def register_scalar(opname, fobject, mtype, connection, Connection, dynamic=False):

    ret = None
    literal = 0
    if mtype.upper() == "STRING":
        ret = "NULL, NULL, result->data"
    elif mtype.upper() == "INT":
        ret = "NULL,result->data, NULL"
    elif mtype.upper() == "FLOAT":
        ret = "result->data, NULL, NULL"
    elif mtype.upper() == "STRINGLITERAL":
        mtype = "STRING"
        ret = "NULL, NULL, result->data"
        literal = 1
    module_name = fobject.__module__
    importstring = module_name + "." + opname
    parts = importstring.split(".")
    importstring = ".".join(parts[-3:])
    wrapperlib = "libwrappedudfs2.so"
    if hasattr(fobject, "jit") and fobject.jit == False:
        wrapperlib = "libwrappedudfs.so"
    if dynamic:
        opname = opname + "_" + "_".join([x.upper() for x in mtype.split(", ")])
    createfunction = """
        CREATE or replace FUNCTION {}(*)
        RETURNS {}
        LANGUAGE C {{
            char* funcn= "{}";
            char* funct = "scalar";
            #pragma CFLAGS -I$CURRENT/YeSQL/functions
            #include "udfs.h"
            #include <stdio.h>
            #include <stdlib.h>
            #include <dlfcn.h>
            #include <link.h>
            void* py_handle = dlopen("/home/johnfouf/yesql_repo/yesql/YeSQL/functions/{}", RTLD_LAZY | RTLD_GLOBAL | RTLD_DEEPBIND);
            if (!py_handle) {{
                fprintf(stderr, "dlmopen failed: %s\\n", dlerror());
                return NULL;
            }}

            // Παίρνουμε pointer στη συνάρτηση udfwrapper
            int (*udf)(char*, int, ArrayInfo1*, double*, int*, char**, int, int, char**) =
                (int (*)(char*, int, ArrayInfo1*, double*, int*, char**, int, int, char**)) dlsym(py_handle, "udfwrapper");

            if (!udf) {{
                fprintf(stderr, "dlsym failed: %s\\n", dlerror());
                dlclose(py_handle);
                return NULL;
            }}

            int count = 1;
            char* funcname = "{}";

            if (paramsnum != 0)
                for (int i = 0; i < paramsnum; ++i) {{
                    int tempcount = ((struct cudf_data_struct_str*)__inputs[i])->count;
                    if (tempcount > 1) {{
                        count = tempcount;
                        break;
                    }}
                }}

            result->initialize(result, count);
            ArrayInfo1 arrays[paramsnum];

            for (int i = 0; i < paramsnum; ++i) {{
                arrays[i].type = ((struct cudf_data_struct_str*)__inputs[i])->datatype;
                arrays[i].array = ((struct cudf_data_struct_str*)__inputs[i])->data;
                arrays[i].size = ((struct cudf_data_struct_str*)__inputs[i])->count;

                if (arrays[i].size == 1) {{
                    arrays[i].size = count;
                    int N = count;
                    int element_size = 0;

                    if (arrays[i].type == 0) element_size = sizeof(char*);
                    else if (arrays[i].type == 1) element_size = sizeof(int);
                    else if (arrays[i].type == 2) element_size = sizeof(double);
                    else if (arrays[i].type == 4) element_size = sizeof(int8_t);
                    else if (arrays[i].type == 5) element_size = sizeof(int16_t);

                    void* new_array = malloc(element_size * count);

                    if (arrays[i].type == 0) {{
                        void** str_array = (void**)new_array;
                        void* str_ptr = *((void**)arrays[i].array);
                        for (int k = 0; k < N; ++k) {{
                            str_array[k] = str_ptr;
                        }}
                    }} else {{
                        for (int k = 0; k < N; ++k) {{
                            memcpy((char*)new_array + k * element_size, arrays[i].array, element_size);
                        }}
                    }}

                    arrays[i].array = new_array;
                }}
            }}

            char* error_message = NULL;
            int ret = udf(funcname, paramsnum, arrays, {}, count, {}, &error_message);
            if (ret == 0)
                return 0;
            else
                return error_message;

            // Κλείσιμο του handle αν θες να μη μείνει ανοιχτό
            dlclose(py_handle);
        }};
    """.format(
        opname, mtype, importstring, wrapperlib, importstring, ret, literal
    )

    if opname not in udfs_cache:
        try:
            connection.cursor().executetrace(createfunction)
        except:
            connection.executetrace(createfunction)
    elif udfs_cache[opname] != createfunction:
        try:
            connection.cursor().executetrace(createfunction)
        except:
            connection.executetrace(createfunction)
    udfs_cache[opname] = createfunction
    if opname == "get_debug":
        createfunction = createfunction.replace(
            "libwrappedudfs2.so", "libwrappedudfs.so"
        )
        connection.cursor().executetrace(createfunction)


def register_table(
    opname,
    staticopname,
    fobject,
    mtype,
    datatypes,
    connection,
    Connection,
    dynamic=False,
    extraparams="NULL",
):
    ret = None
    literal = 0
    if mtype == "STRING":
        ret = "NULL, NULL, result->data"
    elif mtype == "INT":
        ret = "NULL,result->data, NULL"
    elif mtype == "float":
        ret = "result->data, NULL, NULL"
    elif mtype == "STRINGLITERAL":
        mtype = "STRING"
        ret = "NULL, NULL, result->data"
        literal = 1
    module_name = fobject.__module__
    importstring = module_name + "." + staticopname
    parts = importstring.split(".")
    importstring = ".".join(parts[-3:])
    initcode = "\n            ".join(
        [
            staticopname
            + "_"
            + str(n)
            + "->initialize("
            + staticopname
            + "_"
            + str(n)
            + ", arraysresult[0].size);"
            for n in range(len(datatypes))
        ]
    )
    copycode = "\n                ".join(
        [
            staticopname
            + "_"
            + str(n)
            + "->data[k] = (("
            + datatypes[n]
            + ")arraysresult["
            + str(n)
            + "].array)[k];"
            for n in range(len(datatypes))
        ]
    )
    freecode = "\n            ".join(
        ["free(arraysresult[" + str(n) + "].array);" for n in range(len(datatypes))]
    )
    if extraparams != "NULL":
        extraparams = f'"{extraparams}"'
    if dynamic:
        opname = opname + "_" + "_".join([x.upper() for x in mtype.split(", ")])
    wrapfunction = "tableudfwrapper"
    # if opname == 'myfilt':
    if "<class '_cffi_backend._CDataBase'>" in [
        str(annotation)
        for parameter, annotation in fobject.__annotations__.items()
        if parameter != "return"
    ]:
        wrapfunction = "systemwrapper"
    if "<class 'numpy.ndarray'>" in [
        str(annotation)
        for parameter, annotation in fobject.__annotations__.items()
        if parameter != "return"
    ]:
        wrapfunction = "numpyudfwrapper"
    wrapperlib = "libwrappedudfs2.so"
    if hasattr(fobject, "jit") and fobject.jit == False:
        wrapperlib = "libwrappedudfs.so"
    createfunction = """
        CREATE OR REPLACE FUNCTION {}(*)
        RETURNS TABLE ({})
        LANGUAGE C
        {{char* funcn = "{}"; char* funct = "table";

            #pragma CFLAGS -I$CURRENT/YeSQL/functions
            #include "udfs.h"
            #include <stdio.h>
            #include <stdlib.h>
            #include <dlfcn.h>
            #include <link.h>
            void* py_handle = dlopen("/home/johnfouf/yesql_repo/yesql/YeSQL/functions/{}", RTLD_LAZY | RTLD_GLOBAL | RTLD_DEEPBIND);
            if (!py_handle) {{
                fprintf(stderr, "dlmopen failed: %s\\n", dlerror());
                return NULL;
            }}
            int (*tableudfwrapper)(char*, int, ArrayInfo1*, int, ArrayInfo1*, int, char*, char**) = (int (*)(char*, int, ArrayInfo1*, int, ArrayInfo1*, int, char*, char**)) dlsym(py_handle, "{}");

            ArrayInfo arrays[paramsnum];
            ArrayInfo arraysresult[outputsnum];

            for (int i = 0; i < paramsnum; ++i) {{
                 arrays[i].type = ((struct cudf_data_struct_str*)__inputs[i])->datatype;
                 arrays[i].array = ((struct cudf_data_struct_str*)__inputs[i])->data;
                 arrays[i].size = ((struct cudf_data_struct_str*)__inputs[i])->count;
            }}
            for (int i = 0; i < outputsnum; ++i) {{
                 arraysresult[i].type = ((struct cudf_data_struct_str*)__outputs[i])->datatype;
            }}

            char* lala = "{}";
            int count = 0;
            if (paramsnum != 0)
                count = ((struct cudf_data_struct_str*)__inputs[0])->count;
            char* error_message = NULL;
            tableudfwrapper(lala, paramsnum, arrays, outputsnum, arraysresult, count, {}, &error_message);
            if (error_message!=NULL){{
                 {}
                 free(arrays);
                 return error_message;
             }}
            {}
            for (int k=0; k<arraysresult[0].size; k++){{
                {}
            }}
            {}
            freecarray(arrays, outputsnum, arraysresult[0].size);
            free(arrays);
            free(arraysresult);
        }};
    """.format(
        opname,
        mtype,
        importstring,
        wrapperlib,
        wrapfunction,
        importstring,
        extraparams,
        freecode,
        initcode,
        copycode,
        freecode,
    )
    # print(createfunction)
    if opname not in udfs_cache:
        try:
            connection.cursor().executetrace(createfunction)
        except:
            connection.executetrace(createfunction)
    elif udfs_cache[opname] != createfunction:
        try:
            connection.cursor().executetrace(createfunction)
        except:
            connection.executetrace(createfunction)
    udfs_cache[opname] = createfunction


def register_aggregate(opname, fobject, mtype, connection, Connection, dynamic=False):
    ret = None
    global createfunctions
    if mtype == "STRING":
        ret = "NULL, NULL, result->data"
    elif mtype == "INT":
        ret = "NULL,result->data, NULL"
    elif mtype == "FLOAT":
        ret = "result->data, NULL, NULL"
    module_name = fobject.__module__
    importstring = module_name + "." + opname
    parts = importstring.split(".")
    importstring = ".".join(parts[-3:])

    if dynamic:
        opname = opname + "_" + "_".join([x.upper() for x in mtype.split(", ")])
    wrapperlib = "libwrappedudfs2.so"
    if hasattr(fobject, "jit") and fobject.jit == False:
        wrapperlib = "libwrappedudfs.so"
    createfunction = """
CREATE or replace AGGREGATE {}(*)
RETURNS {}
LANGUAGE C {{char* funcn= "{}"; char* funct = "aggregate";
             #pragma CFLAGS -I$CURRENT/YeSQL/functions
            #include "udfs.h"
            #include <stdio.h>
            #include <stdlib.h>
            #include <dlfcn.h>
            #include <link.h>
            #include <string.h>
            char* funcname = "{}";
            void* py_handle = dlopen("/home/johnfouf/yesql_repo/yesql/YeSQL/functions/{}", RTLD_LAZY | RTLD_GLOBAL | RTLD_DEEPBIND);
            if (!py_handle) {{
                fprintf(stderr, "dlmopen failed: %s\\n", dlerror());
                return NULL;
            }}

int (*aggregatewrapper)(char*, int, int, ArrayInfo1*, double*, int*, char**, int, size_t*, char**) =
        (int (*)(char*, int, int, ArrayInfo1*, double*, int*, char**, int, size_t*, char**)) dlsym(py_handle, "aggregatewrapper");

int loc_paramsnum = paramsnum;
int groupby = 0;
int aggr_group_count = aggr_group.count;
if ((((struct cudf_data_struct_oid*)__inputs[paramsnum-1])->count != ((struct cudf_data_struct_str*)__inputs[0])->count)) {{
//if (aggr_group_count!=1 )
    groupby = 1;
    loc_paramsnum = paramsnum - 1;
}} else if ((((struct cudf_data_struct_oid*)__inputs[paramsnum-1])->data[0] == 0)) {{
    groupby = 1;
    loc_paramsnum = paramsnum - 1;
}} else {{
    aggr_group_count = 1;
}}

int input_count = ((struct cudf_data_struct_str*)__inputs[0])->count;
size_t* count_per_group = (size_t*) malloc(sizeof(size_t) * aggr_group_count);
memset(count_per_group, 0, sizeof(size_t) * aggr_group_count);
size_t* idx_per_group = (size_t*) malloc(sizeof(size_t) * aggr_group_count);
ArrayInfo arrays[loc_paramsnum];

size_t* group_ids = groupby ? ((struct cudf_data_struct_oid*)__inputs[loc_paramsnum])->data : NULL;

for (size_t i = 0; i < input_count; i++) {{
    size_t grp = groupby ? group_ids[i] : 0;
    ++count_per_group[grp];
}}
//result->initialize(result, 1); //aggr_group_count);
result->initialize(result,aggr_group_count);
for (int k = 0; k < loc_paramsnum; ++k) {{
    memset(idx_per_group, 0, sizeof(size_t) * aggr_group_count);
    arrays[k].type = ((struct cudf_data_struct_str*)__inputs[k])->datatype;
    size_t total_count = 0;
    for (size_t i = 0; i < aggr_group_count; ++i) total_count += count_per_group[i];
      if (arrays[k].type == 0) {{
        char** flat_data = (char**) malloc(sizeof(char*) * total_count);
        char*** mat_input = (char***) malloc(sizeof(char**) * aggr_group_count);
        size_t offset = 0;
        for (size_t i = 0; i < aggr_group_count; i++) {{
            mat_input[i] = &flat_data[offset];
            offset += count_per_group[i];
        }}
        char** input_data = ((struct cudf_data_struct_str*)__inputs[k])->data;
        for (size_t i = 0; i < input_count; i++) {{
            size_t grp = groupby ? group_ids[i] : 0;
            mat_input[grp][idx_per_group[grp]++] = input_data[i];
        }}
        arrays[k].array = mat_input;
      }} else if (arrays[k].type == 1) {{
        int* flat_data = (int*) malloc(sizeof(int) * total_count);
        int** mat_input = (int**) malloc(sizeof(int*) * aggr_group_count);
        size_t offset = 0;
        for (size_t i = 0; i < aggr_group_count; i++) {{
            mat_input[i] = &flat_data[offset];
            offset += count_per_group[i];
        }}
        int* input_data = ((struct cudf_data_struct_int*)__inputs[k])->data;
        for (size_t i = 0; i < input_count; i++) {{
            size_t grp = groupby ? group_ids[i] : 0;
            mat_input[grp][idx_per_group[grp]++] = input_data[i];
        }}
        arrays[k].array = mat_input;
      }} else if (arrays[k].type == 2) {{
        double* flat_data = (double*) malloc(sizeof(double) * total_count);
        double** mat_input = (double**) malloc(sizeof(double*) * aggr_group_count);
        size_t offset = 0;
        for (size_t i = 0; i < aggr_group_count; i++) {{
            mat_input[i] = &flat_data[offset];
            offset += count_per_group[i];
        }}
        double* input_data = ((struct cudf_data_struct_dbl*)__inputs[k])->data;
        for (size_t i = 0; i < input_count; i++) {{
            size_t grp = groupby ? group_ids[i] : 0;
            mat_input[grp][idx_per_group[grp]++] = input_data[i];
        }}
        arrays[k].array = mat_input;
    }}

    arrays[k].size = input_count;
}}

if (groupby)
    paramsnum = paramsnum - 1;

char* error_message = NULL;
int ret = aggregatewrapper(funcname, input_count, paramsnum, arrays, {}, aggr_group_count, count_per_group, &error_message);
if (ret == 0)
    return 0;
else
    return error_message;

}};
    """.format(
        opname, mtype, importstring, importstring, wrapperlib, ret
    )

    if opname not in udfs_cache:
        try:
            connection.cursor().executetrace(createfunction)
        except:
            connection.executetrace(createfunction)
    elif udfs_cache[opname] != createfunction:
        try:
            connection.cursor().executetrace(createfunction)
        except:
            connection.executetrace(createfunction)
    udfs_cache[opname] = createfunction


# parses the query plan that uses the dummy function funcexpand to get the datatypes
def parseplan(ss, con, multiset_functions):
    subquery = "explain select * from expand((" + ss + ")) as vname;"
    countrows = con.executetrace(subquery)
    listbats = []
    lines = []
    fname = []
    fcalls = []
    fname_returnx = []
    for i in range(countrows):
        nextline = con.next()[0]
        if (
            '"{\\n            i->initialize(i, 1);\\n            i->data[0] =  \\"4b41025d848e3c0e54f6dbd07fde1777\\"'
            in nextline
        ):
            lll = re.findall(
                r"(?:(X_\d+):bat\[:(\w+)\])|(?:(X_\d+):(\w+))", nextline, flags=0
            )[1:]
            yyy = []
            lines.append([tuple(item for item in tpl if item != "") for tpl in lll])
            break
        elif "algebra.project" in nextline:
            pattern = "|".join(["\\b" + x + "\\b" for x in fname_returnx])
            match = re.search(r"(" + pattern + ")", nextline)
            if match is not None and match.groups(1)[0].strip() != "":
                fname_returnx[fname_returnx.index(match.groups(1)[0].strip())] = (
                    re.findall(r"((?:X|C)_\d+)", nextline, flags=0)[0]
                )
        elif (
            "capi.eval" in nextline
            or "capi.subeval_aggr" in nextline
            or "capi.eval_aggr" in nextline
        ):
            if (
                "batcapi.eval" in nextline
                or "capi.subeval_aggr" in nextline
                or "capi.eval_aggr" in nextline
            ):
                match = re.search(r'char\*\sfuncn= \\"\w+\.\w+\.(\w+)\\";', nextline)
                match2 = re.search(r'char\*\sfuncn= \\"(\w+\.\w+\.\w+)\\";', nextline)
                if match:
                    if match.groups(1)[0] in multiset_functions:
                        fname.append(match.groups(1)[0])
                        fcalls.append(match2.groups(1)[0])
                        fname_returnx.append(
                            re.findall(r"((?:X|C)_\d+)", nextline, flags=0)[0]
                        )
            else:
                match = re.search(r'char\*\sfuncn= \\"\w+\.\w+\.(\w+)\\";', nextline)
                match2 = re.search(r'char\*\sfuncn= \\"(\w+\.\w+\.\w+)\\";', nextline)
                if match:
                    if match.groups(1)[0] in multiset_functions:
                        fname.append(match.groups(1)[0])
                        fcalls.append(match2.groups(1)[0])
                        fname_returnx.append(
                            re.findall(r"((?:X|C)_\d+)", nextline, flags=0)[0]
                        )
                # raise Exception('Multiset Functions must have at least one bat input and not only scalars')
    i = 0
    while i < len(fname_returnx):
        if fname_returnx[i] not in [t[0] for sublist in lines for t in sublist]:
            del fname_returnx[i]
            del fname[i]
            del fcalls[i]
            i = i - 1
        i += 1
    return lines, fname_returnx, fname, fcalls


# this function is complementary to the above. Expand function is registered again using the correct datatypes that are infered from the plan
def register_expand(
    con,
    expanddatatypes,
    multisetindex,
    vname,
    fnames,
    fobject,
    udfargs,
    paramscount,
    aggr=True,
):
    ret = None
    wrapperlib = "libwrappedudfs2.so"
    # fobject = self.multiset_functions[fnames[0]]
    if hasattr(fobject, "jit") and fobject.jit == False:
        wrapperlib = "libwrappedudfs.so"
        print("kaka")
    if hasattr(fobject, "jit") and fobject.jit == True:
        print("fofo")
        wrapperlib = "libwrappedudfs2.so"

    if paramscount == 0:
        udfargs = []
    paramsnum = ",".join(
        ["v" + str(x + 1) + " " + y for x, y in enumerate(expanddatatypes)]
    )
    expname = "_".join([x for x in expanddatatypes])
    udfar = "".join([str(x) for x in udfargs])
    inits = "\n            ".join(
        [
            "v"
            + str(n + 1)
            + "->initialize(v"
            + str(n + 1)
            + ", arraysresult[0].size);"
            for n in range(len(expanddatatypes))
        ]
    )
    assignments = "\n               ".join(
        [
            "v"
            + str(n + 1)
            + "->data[k] = (("
            + datatypemap[expanddatatypes[n]]
            + ")arraysresult["
            + str(n)
            + "].array)[k];"
            for n in range(len(expanddatatypes))
        ]
    )
    frees = "\n            ".join(
        [
            "free(arraysresult[" + str(n) + "].array);"
            for n in range(len(expanddatatypes))
        ]
    )
    multisetc = ",".join([str(x) for x in multisetindex])
    multisetindexloop = "\n                 ".join(
        [
            "multisetindex_to_c[" + str(i) + "] =" + str(multisetindex[i]) + ";"
            for i in range(len(multisetindex))
        ]
    )
    udfargsloop = "\n                 ".join(
        [
            "udfargs_to_c[" + str(i) + "] =" + str(udfargs[i]) + ";"
            for i in range(len(udfargs))
        ]
    )
    inputcount = "arg2.count" if paramscount > 0 else "0"
    expandwrapper = "expandwrapper"
    if aggr:
        expandwrapper = "expandaggrwrapper"
    # wrapperlib = "libwrappedudfs2.so"
    # if (hasattr(fobject, 'jit') and fobject.jit == False):
    #     wrapperlib = "libwrappedudfs.so"
    createfunction = (
        """
            CREATE or replace FUNCTION expand_"""
        + fnames[0].split(".")[-1]
        + expname
        + udfar
        + """(*)
            RETURNS TABLE ("""
        + paramsnum
        + """)
            LANGUAGE C {{
            #pragma CFLAGS -I$CURRENT/YeSQL/functions
            #include "udfs.h"
            #include <stdio.h>
            #include <stdlib.h>
            #include <dlfcn.h>
            #include <link.h>
            void* py_handle = dlopen("/home/johnfouf/yesql_repo/yesql/YeSQL/functions/"""
        + wrapperlib
        + """", RTLD_LAZY | RTLD_GLOBAL | RTLD_DEEPBIND);
            if (!py_handle) {{
                fprintf(stderr, "dlmopen failed: %s\\n", dlerror());
                return NULL;
            }}
            int (*"""
        + expandwrapper
        + ''')(char*, int, ArrayInfo1*, int, ArrayInfo1*, int*, int, int, int*, char**) =  (int (*)(char*, int, ArrayInfo1*, int, ArrayInfo1*, int*, int, int, int*, char**)) dlsym(py_handle, "'''
        + expandwrapper
        + """");

            ArrayInfo arrays[paramsnum];
            ArrayInfo arraysresult[outputsnum];
            int * udfargs_to_c = (int *)malloc("""
        + str(len(udfargs))
        + """ * sizeof(int));
                 """
        + udfargsloop
        + """
            int * multisetindex_to_c = (int *)malloc("""
        + str(len(multisetindex))
        + """ * sizeof(int));
                 """
        + multisetindexloop
        + '''
            for (int i = 0; i < paramsnum; ++i) {{
                 arrays[i].type = ((struct cudf_data_struct_str*)__inputs[i])->datatype;
                 arrays[i].array = ((struct cudf_data_struct_str*)__inputs[i])->data;
                 arrays[i].size = ((struct cudf_data_struct_str*)__inputs[i])->count;
            }}
            for (int i = 0; i < outputsnum; ++i) {{
                 arraysresult[i].type = ((struct cudf_data_struct_str*)__outputs[i])->datatype;
                 //arraysresult[i].size = arg2.count;
            }}
            char* expname = "'''
        + fnames[0]
        + """";
            char* errormessage = NULL;
            int ret = """
        + expandwrapper
        + """(expname, paramsnum, arrays, outputsnum, arraysresult, multisetindex_to_c,"""
        + str(len(multisetindex))
        + """ , """
        + inputcount
        + """, udfargs_to_c, &errormessage);
            if (errormessage!=NULL){{
                 """
        + frees
        + """
                 free(multisetindex_to_c);
                 free(udfargs_to_c);
                 free(arrays);
                 free(arraysresult);
                 return errormessage;
            }}

            """
        + inits
        + """
            char* input_string;
            size_t len;
            for (int k=0; k<arraysresult[0].size; k++){{
                 """
        + assignments
        + """
            }}
            freecarray(arraysresult, outputsnum, arraysresult[0].size);
            free(multisetindex_to_c);
            free(udfargs_to_c);
            free(arrays);

            return 0;
           }};
    """
    )
    con.executetrace(createfunction)
