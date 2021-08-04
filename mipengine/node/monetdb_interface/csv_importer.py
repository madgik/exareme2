import csv
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict
from typing import List

from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import Float
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import create_engine
from sqlalchemy import null
from sqlalchemy.exc import OperationalError

from mipengine.common_data_elements import CommonDataElement
from mipengine.common_data_elements import CommonDataElements
from mipengine.node import DATA_TABLE_PRIMARY_KEY

AMOUNT_OF_ROWS_TO_INSERT_INTO_SQL_PER_CALL = 100


def create_pathology_metadata_table(
    pathology: str, pathology_common_data_elements: Dict[str, CommonDataElement]
):
    metadata_table_name = pathology + "_metadata"
    metadata_table = Table(
        metadata_table_name,
        db_engine_metadata,
        Column("code", String(100), primary_key=True),
        Column("label", String(255)),
        Column("sql_type", String(10)),
        Column("categorical", Boolean),
        Column("enumerations", String(255)),
        Column("min", Integer),
        Column("max", Integer),
    )
    db_engine_metadata.drop_all(db_engine, checkfirst=True, tables=[metadata_table])
    db_engine_metadata.create_all(db_engine, tables=[metadata_table])

    for (
        common_data_element_code,
        common_data_element,
    ) in pathology_common_data_elements.items():
        # Parse the special values (Optional, Enumerations) to sql format
        if common_data_element.enumerations is not None:
            enumerations_sql_value = ", ".join(
                [str(e) for e in common_data_element.enumerations]
            )
        else:
            enumerations_sql_value = null()

        if common_data_element.min is not None:
            min_sql_value = common_data_element.min
        else:
            min_sql_value = null()

        if common_data_element.max is not None:
            max_sql_value = common_data_element.max
        else:
            max_sql_value = null()

        cde_values = {
            "code": common_data_element_code,
            "label": common_data_element.label,
            "sql_type": common_data_element.sql_type,
            "categorical": common_data_element.is_categorical,
            "enumerations": enumerations_sql_value,
            "min": min_sql_value,
            "max": max_sql_value,
        }

        insert_query = metadata_table.insert().values(cde_values)
        db_engine.execute(insert_query, cde_values)


def convert_sql_type_to_monetdb_type(sql_type: str):
    """Converts metadata sql type to monetdb sqlalchemy type
    int -> Integer
    real -> Float
    text -> String(100)
    """
    return {"int": Integer, "real": Float, "text": String(100)}[str.lower(sql_type)]


def create_pathology_data_table(
    pathology: str, pathology_common_data_elements: Dict[str, CommonDataElement]
):
    column_names = [
        cde_code for cde_code, cde in pathology_common_data_elements.items()
    ]
    column_types = [
        convert_sql_type_to_monetdb_type(cde.sql_type)
        for cde_code, cde in pathology_common_data_elements.items()
    ]
    columns = [
        Column(column_name.lower(), column_type)
        for column_name, column_type in zip(column_names, column_types)
    ]

    # The row_id column, the primary key of the table, it's not part of the metadata
    row_id_column = Column(
        DATA_TABLE_PRIMARY_KEY,
        convert_sql_type_to_monetdb_type("int"),
        primary_key=True,
        autoincrement=True,
    )
    columns.append(row_id_column)

    data_table_name = pathology + "_data"
    data_table = Table(data_table_name, db_engine_metadata, *columns)

    db_engine_metadata.drop_all(db_engine, checkfirst=True, tables=[data_table])
    db_engine_metadata.create_all(db_engine, tables=[data_table])


def import_dataset_csv_into_data_table(
    csv_file_path: Path,
    pathology: str,
    pathology_common_data_elements: Dict[str, CommonDataElement],
):
    data_table_name = pathology + "_data"

    # Open the csv
    dataset_csv_content = open(csv_file_path, "r", encoding="utf-8")
    dataset_csv_reader = csv.reader(dataset_csv_content)

    # Validate that all columns exist
    csv_header = next(dataset_csv_reader)
    for column in csv_header:
        if column not in pathology_common_data_elements.keys():
            raise KeyError("Column " + column + " does not exist in the metadata!")

    # Create the prefix for the INSERT statement ("INSERT INTO ... ( COLUMN_A, ... )
    insert_query_columns = ",".join(csv_header)
    insert_query_prefix = (
        f"INSERT INTO {data_table_name} ({insert_query_columns}) VALUES "
    )

    # Insert data
    row_counter = 0
    bulk_insert_query_values = "("
    for row in dataset_csv_reader:
        row_counter += 1

        for (value, column) in zip(row, csv_header):
            # Validate the value enumerations
            # column_enumerations = pathology_common_data_elements[column].enumerations
            # # print(f"column_enumerations-> {column_enumerations}")
            # if column_enumerations and value and value not in column_enumerations:
            #     breakpoint()
            #     if value not in [str(c) for c in column_enumerations]:
            #         breakpoint()
            #         raise ValueError(f"Value {value} in column {column} does not "
            #                          f"have one of the allowed enumerations: {column_enumerations}")

            # Validate the value, min limit
            column_min_value = pathology_common_data_elements[column].min
            if column_min_value and value and value < column_min_value:
                raise ValueError(
                    f"Value {value} in column {column} should be less than {column_min_value}"
                )

            # Validate the value, max limit
            column_max_value = pathology_common_data_elements[column].max
            if column_max_value and value and value > column_max_value:
                raise ValueError(
                    f"Value {value} in column {column} should be greater than {column_max_value}"
                )

            # Add the value to the insert query
            if value.strip() == "":
                bulk_insert_query_values += "null, "
            elif pathology_common_data_elements[column].sql_type == "text":
                bulk_insert_query_values += "'" + value.strip() + "', "
            else:
                bulk_insert_query_values += value.strip() + ", "

        # Execute insert query every AMOUNT_OF_ROWS_TO_INSERT_INTO_SQL_PER_CALL rows
        if row_counter % AMOUNT_OF_ROWS_TO_INSERT_INTO_SQL_PER_CALL == 0:
            bulk_insert_query_values = bulk_insert_query_values[:-2]
            bulk_insert_query_values += ");"

            try:
                db_engine.execute(insert_query_prefix + bulk_insert_query_values)
            except OperationalError:
                find_error_on_bulk_insert_query(
                    data_table_name,
                    bulk_insert_query_values,
                    csv_header,
                    pathology_common_data_elements,
                    csv_file_path,
                )
                raise ValueError("Error inserting the CSV to the database.")

            bulk_insert_query_values = "("
        else:
            bulk_insert_query_values = bulk_insert_query_values[:-2]
            bulk_insert_query_values += "),("

    # Insertion of the last rows
    if row_counter % AMOUNT_OF_ROWS_TO_INSERT_INTO_SQL_PER_CALL != 0:
        bulk_insert_query_values = bulk_insert_query_values[:-3]
        bulk_insert_query_values += ");"

        try:
            db_engine.execute(insert_query_prefix + bulk_insert_query_values)
        except OperationalError:
            find_error_on_bulk_insert_query(
                data_table_name,
                bulk_insert_query_values,
                csv_header,
                pathology_common_data_elements,
                csv_file_path,
            )


def find_error_on_bulk_insert_query(
    data_table_name: str,
    bulk_insert_query: str,
    csv_header: List[str],
    pathology_common_data_elements: Dict[str, CommonDataElement],
    csv_file_path: Path,
):
    # Removing the first and last parenthesis
    bulk_insert_query = bulk_insert_query[1:-2]
    # Removing the ' from character values
    bulk_insert_query = bulk_insert_query.replace("'", "")
    # Call findErrorOnSqlQuery for each row in the bulk query
    for row in bulk_insert_query.split("),("):
        find_error_on_sql_query(
            data_table_name,
            row.split(","),
            csv_header,
            pathology_common_data_elements,
            csv_file_path,
        )
    raise ValueError(
        f"""Error inserting into the database,
        while inserting csv: {csv_file_path}
        """
    )


def find_error_on_sql_query(
    data_table_name: str,
    row_values: List[str],
    csv_header: List[str],
    pathology_common_data_elements: Dict[str, CommonDataElement],
    csv_file_path: Path,
):
    # Insert the code column into the database
    # and then update it for each row to find where the problem is
    if csv_header[0] != "subjectcode":
        raise ValueError(
            f"""Error inserting into the database,
                the csv: {csv_file_path} ,
                subjectcode is not the first column.
                """
        )

    subject_code = row_values[0]
    insert_query = (
        f"INSERT INTO {data_table_name} (subjectcode) VALUES ('{subject_code}')"
    )
    db_engine.execute(insert_query)

    for (value, column) in zip(row_values[1:], csv_header[1:]):
        if pathology_common_data_elements[column].sql_type == "text":
            update_query = (
                f"UPDATE {data_table_name} SET {column} = '{value.strip()}' "
                f"WHERE subjectcode = '{subject_code}'"
            )
        elif value.strip() == "":
            update_query = f"UPDATE {data_table_name} SET {column} = null WHERE subjectcode = '{subject_code}'"
        else:
            update_query = (
                f"UPDATE {data_table_name} SET {column} = {value.strip()} "
                f"WHERE subjectcode = '{subject_code}'"
            )

        try:
            db_engine.execute(update_query)
        except OperationalError:
            raise ValueError(
                f"""Error inserting into the database.
                Could not insert value: '{value.strip()}',
                into column: '{column}',
                at row with subjectcode: {subject_code},
                while inserting csv: {csv_file_path}
                """
            )


parser = ArgumentParser()
parser.add_argument(
    "-folder",
    "--pathologies_folder_path",
    required=True,
    help="The folder with the pathologies data.",
)
parser.add_argument(
    "-spec",
    "--specific_pathologies",
    required=False,
    help="Specific pathologies to parse.",
)
parser.add_argument(
    "-user", "--monetdb_username", required=True, help="MonetDB username."
)
parser.add_argument(
    "-pass", "--monetdb_password", required=True, help="MonetDB password."
)
parser.add_argument("-url", "--monetdb_url", required=True, help="MonetDB url.")
parser.add_argument("-farm", "--monetdb_farm", required=True, help="MonetDB farm.")

args = parser.parse_args()
data_path = args.pathologies_folder_path
pathologies_to_parse = args.specific_pathologies
monetdb_username = args.monetdb_username
monetdb_password = args.monetdb_password
monetdb_url = args.monetdb_url
monetdb_farm = args.monetdb_farm

print(f"Importing metadata of pathologies in {data_path}")
common_data_elements = CommonDataElements(Path(data_path))

db_engine_metadata = MetaData()
db_engine = create_engine(
    f"monetdb://{monetdb_username}:{monetdb_password}@" f"{monetdb_url}/{monetdb_farm}:"
)

data_abs_path = os.path.abspath(data_path)
pathology_names = next(os.walk(data_abs_path))[1]

if pathologies_to_parse is not None:
    pathologies_to_convert = pathologies_to_parse.split(",")
    pathology_names = [
        pathology_name
        for pathology_name in pathology_names
        if pathology_name in pathologies_to_convert
    ]
print("Importing CSVs for pathologies: " + ",".join(pathology_names))

# Import all pathologies
for pathology_name in pathology_names:
    create_pathology_metadata_table(
        pathology_name, common_data_elements.pathologies[pathology_name]
    )

    create_pathology_data_table(
        pathology_name, common_data_elements.pathologies[pathology_name]
    )

    # Import each csv of the pathology
    pathology_folder_path = Path(os.path.join(data_abs_path, pathology_name))
    for csv_path in pathology_folder_path.glob("*.csv"):
        print(f"Importing CSV: {csv_path}")
        import_dataset_csv_into_data_table(
            csv_path, pathology_name, common_data_elements.pathologies[pathology_name]
        )
