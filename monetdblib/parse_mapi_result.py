import pymonetdb
from . import mapi_async as mapi
from collections import namedtuple
from pymonetdb.sql import monetize, pythonize

Description = namedtuple('Description', ('name', 'type_code', 'display_size', 'internal_size', 'precision', 'scale',
                                         'null_ok'))
def _parse_tuple(line, description):
        """
        parses a mapi data tuple, and returns a list of python types
        """
        elements = line[1:-1].split(',\t')
        if len(elements) == len(description):
            return tuple([pythonize.convert(element.strip(), description[1])
                          for (element, description) in zip(elements,
                                                            description)])
       
def parse(block):
        """ parses the mapi result into a resultset"""

        if not block:
            block = ""

        columns = 0
        column_name = ""
        scale = display_size = internal_size = precision = 0
        null_ok = False
        type_ = []

        for line in block.split("\n"):

            if line.startswith(mapi.MSG_QTABLE):
                _query_id, rowcount, columns, tuples = line[2:].split()[:4]

                columns = int(columns)  # number of columns in result
                rowcount = int(rowcount)  # total number of rows
                # tuples = int(tuples)     # number of rows in this set
                _rows = []

                # set up fields for description
                # table_name = [None] * columns
                column_name = [None] * columns
                type_ = [None] * columns
                display_size = [None] * columns
                internal_size = [None] * columns
                precision = [None] * columns
                scale = [None] * columns
                null_ok = [None] * columns
                # typesizes = [(0, 0)] * columns

                _offset = 0
                lastrowid = None

            elif line.startswith(mapi.MSG_HEADER):
                (data, identity) = line[1:].split("#")
                values = [x.strip() for x in data.split(",")]
                identity = identity.strip()

                if identity == "name":
                    column_name = values
                elif identity == "table_name":
                    _ = values  # not used
                elif identity == "type":
                    type_ = values
                elif identity == "length":
                    _ = values  # not used
                elif identity == "typesizes":
                    typesizes = [[int(j) for j in i.split()] for i in values]
                    internal_size = [x[0] for x in typesizes]
                    for num, typeelem in enumerate(type_):
                        if typeelem in ['decimal']:
                            precision[num] = typesizes[num][0]
                            scale[num] = typesizes[num][1]
                

                description = []
                for i in range(columns):
                    description.append(Description(column_name[i], type_[i], display_size[i], internal_size[i],
                                                   precision[i], scale[i], null_ok[i]))

                _offset = 0
                lastrowid = None
                
            elif line.startswith(mapi.MSG_TUPLE):
                values = _parse_tuple(line,description)
                _rows.append(values)

            elif line.startswith(mapi.MSG_TUPLE_NOSLICE):
                _rows.append((line[1:],))

            elif line.startswith(mapi.MSG_QBLOCK):
                _rows = []

            elif line.startswith(mapi.MSG_QSCHEMA):
                _offset = 0
                lastrowid = None
                _rows = []
                description = None
                rowcount = -1

            elif line.startswith(mapi.MSG_QUPDATE):
                (affected, identity) = line[2:].split()[:2]
                _offset = 0
                _rows = []
                description = None
                rowcount = int(affected)
                lastrowid = int(identity)
                _query_id = -1

            elif line.startswith(mapi.MSG_QTRANS):
                _offset = 0
                lastrowid = None
                _rows = []
                description = None
                rowcount = -1

            elif line == mapi.MSG_PROMPT:
                return _rows

            elif line.startswith(mapi.MSG_ERROR):
                _exception_handler(ProgrammingError, line[1:])
                
                
