from typing import List

import pytest
from pydantic import ValidationError

from exareme2.worker_communication import ColumnDataFloat
from exareme2.worker_communication import ColumnDataInt
from exareme2.worker_communication import ColumnDataStr
from exareme2.worker_communication import ColumnInfo
from exareme2.worker_communication import DType
from exareme2.worker_communication import SMPCTablesInfo
from exareme2.worker_communication import TableData
from exareme2.worker_communication import TableInfo
from exareme2.worker_communication import TableSchema
from exareme2.worker_communication import TableType
from exareme2.worker_communication import WorkerLiteralDTO
from exareme2.worker_communication import WorkerSMPCDTO
from exareme2.worker_communication import WorkerTableDTO
from exareme2.worker_communication import WorkerUDFDTO
from exareme2.worker_communication import WorkerUDFKeyArguments
from exareme2.worker_communication import WorkerUDFPosArguments
from exareme2.worker_communication import WorkerUDFResults
from exareme2.worker_communication import _WorkerUDFDTOType


@pytest.fixture
def acceptable_name():
    return "AcceptableName"


@pytest.fixture
def name_error_str():
    return "Unacceptable Name"


@pytest.fixture
def column_info_data_type():
    return DType.INT


@pytest.fixture
def column_info_data_type_error():
    return 3.4j


# asserts correct parameters in test_column_info
def test_column_info(acceptable_name, column_info_data_type):
    expected_name = "AcceptableName"
    expected_dt = column_info_data_type
    assert expected_name == acceptable_name
    assert expected_dt == column_info_data_type


# raises error for wrong params in ColumnInfo
# @pytest.mark.xfail
def test_column_info_error(name_error_str, column_info_data_type_error):
    with pytest.raises(ValidationError):
        ColumnInfo(name=name_error_str, dtype=column_info_data_type_error)


def test_column_info_immutable():
    info = ColumnInfo(name="name", dtype=DType.FLOAT)
    with pytest.raises(TypeError):
        info.name = "newname"


# asserts correct parameters in test_column_info
def test_table_schema():
    cols = [
        ColumnInfo(name="AcceptableName1", dtype=DType.FLOAT),
        ColumnInfo(name="AcceptableName2", dtype=DType.FLOAT),
    ]
    assert isinstance(cols, List)
    assert isinstance(cols[0], ColumnInfo)
    assert isinstance(cols[1], ColumnInfo)


def test_table_data_with_different_column_types():
    expected_columns = [
        ColumnDataFloat(data=[1.0, None], name="column1"),
        ColumnDataInt(data=[2, None], name="column2"),
        ColumnDataStr(data=["3", None], name="column3"),
        ColumnDataInt(data=[4, 4], name="column4"),
        ColumnDataFloat(data=[5.0, 5.0], name="column5"),
    ]
    data = TableData(
        name="table_name",
        columns=expected_columns,
    )
    assert data.columns == expected_columns
    assert TableData.parse_raw(data.json()) == data


def test_table_schema_immutable():
    schema = TableSchema(
        columns=[
            ColumnInfo(name="layla", dtype=DType.FLOAT),
            ColumnInfo(name="sheila", dtype=DType.FLOAT),
        ]
    )
    with pytest.raises(TypeError):
        schema.columns = [
            ColumnInfo(name="newname", dtype=DType.FLOAT),
            ColumnInfo(name="newname", dtype=DType.FLOAT),
        ]


@pytest.fixture
def table_info_proper_type():
    return TableInfo(
        name="a_b_c_d_e",
        schema_=TableSchema(
            columns=[
                ColumnInfo(name="layla", dtype=DType.FLOAT),
                ColumnInfo(name="sheila", dtype=DType.FLOAT),
            ]
        ),
        type_=TableType.NORMAL,
    )


def test_table_info_type(table_info_proper_type):
    assert isinstance(table_info_proper_type.type_, TableType)


@pytest.fixture
def table_info_data_schema():
    return TableSchema(
        columns=[
            ColumnInfo(name="layla", dtype=DType.FLOAT),
            ColumnInfo(name="sheila", dtype=DType.FLOAT),
        ]
    )


def test_table_info_schema(table_info_data_schema):
    assert isinstance(table_info_data_schema, TableSchema)


@pytest.fixture
def table_info_data_schema_error():
    return TableSchema(
        columns=[
            ColumnInfo(name=name_error_str, dtype=DType.FLOAT),
            ColumnInfo(name="Sheila", dtype=DType.FLOAT),
        ]
    )


# validation check for table_info
def test_table_info_error():
    with pytest.raises(ValidationError):
        TableInfo(
            name=name_error_str,
            schema_=table_info_data_schema_error,
            type_=TableType.NORMAL,
        )


def test_table_info_immutable():
    info = TableInfo(
        name="a_b_c_d_e",
        schema_=TableSchema(
            columns=[
                ColumnInfo(name="layla", dtype=DType.FLOAT),
                ColumnInfo(name="sheila", dtype=DType.FLOAT),
            ],
        ),
        type_=TableType.NORMAL,
    )
    with pytest.raises(TypeError):
        info.name = "newname"


def test_table_info__valid_name():
    info = TableInfo(
        name="NORMAL_workerid_ctxid_cmdid_resid",
        schema_=TableSchema(columns=[]),
        type_=TableType.NORMAL,
    )
    assert info.worker_id == "workerid"
    assert info.context_id == "ctxid"
    assert info.command_id == "cmdid"
    assert info.result_id == "resid"
    assert info.name_without_worker_id == "NORMAL_ctxid_cmdid_resid"


def test_table_info__invalid_name():
    info = TableInfo(
        name="name",
        schema_=TableSchema(columns=[]),
        type_=TableType.NORMAL,
    )
    with pytest.raises(ValueError):
        info.worker_id


def test_table_data_error():
    with pytest.raises(ValidationError):
        TableData(name="foo", columns=34)


def test_table_data_immutable():
    data = TableData(
        name="table",
        columns=[
            ColumnDataFloat(name="layla", data=[9.1]),
            ColumnDataFloat(name="sheila", data=[9.1]),
        ],
    )
    with pytest.raises(TypeError):
        data.name = "newname"


def test_table_data():
    with pytest.raises(ValidationError):
        TableData(
            name="this is not a table name",
            columns="and this is not a list of columns",
        )


def test_udf_dto_instantiation():
    with pytest.raises(ValidationError) as exc:
        WorkerUDFDTO(type="LITERAL", value="this can be anything")
    assert "should not be instantiated." in str(exc)


def test_udf_dtos_immutable():
    argument = WorkerTableDTO(
        value=TableInfo(
            name="a_b_c_d_e",
            schema_=TableSchema(columns=[ColumnInfo(name="test", dtype=DType.INT)]),
            type_=TableType.NORMAL,
        )
    )
    with pytest.raises(TypeError) as exc:
        argument.value = None
    assert "is immutable" in str(exc)

    argument = WorkerLiteralDTO(value=10)
    with pytest.raises(TypeError) as exc:
        argument.value = "new"
    assert "is immutable" in str(exc)

    argument = WorkerSMPCDTO(
        value=SMPCTablesInfo(
            template=TableInfo(
                name="a_b_c_d_e",
                schema_=TableSchema(columns=[ColumnInfo(name="test", dtype=DType.INT)]),
                type_=TableType.NORMAL,
            )
        )
    )
    with pytest.raises(TypeError) as exc:
        argument.value = "new"
    assert "is immutable" in str(exc)


def test_udf_dtos_correct_type():
    argument = WorkerTableDTO(
        value=TableInfo(
            name="a_b_c_d_e",
            schema_=TableSchema(columns=[ColumnInfo(name="test", dtype=DType.INT)]),
            type_=TableType.NORMAL,
        )
    )
    assert argument.type == _WorkerUDFDTOType.TABLE

    argument = WorkerLiteralDTO(value=10)
    assert argument.type == _WorkerUDFDTOType.LITERAL

    argument = WorkerSMPCDTO(
        value=SMPCTablesInfo(
            template=TableInfo(
                name="a_b_c_d_e",
                schema_=TableSchema(columns=[ColumnInfo(name="test", dtype=DType.INT)]),
                type_=TableType.NORMAL,
            )
        )
    )
    assert argument.type == _WorkerUDFDTOType.SMPC


def get_udf_args_cases():
    return [
        [
            WorkerTableDTO(
                value=TableInfo(
                    name="a_b_c_d_e",
                    schema_=TableSchema(
                        columns=[ColumnInfo(name="test", dtype=DType.INT)]
                    ),
                    type_=TableType.NORMAL,
                )
            ),
        ],
        [
            WorkerTableDTO(
                value=TableInfo(
                    name="a_b_c_d_e",
                    schema_=TableSchema(
                        columns=[ColumnInfo(name="test", dtype=DType.INT)]
                    ),
                    type_=TableType.NORMAL,
                )
            ),
            WorkerLiteralDTO(value="a_b_c_d_e"),
        ],
        [
            WorkerTableDTO(
                value=TableInfo(
                    name="a_b_c_d_e",
                    schema_=TableSchema(
                        columns=[ColumnInfo(name="test", dtype=DType.INT)]
                    ),
                    type_=TableType.NORMAL,
                )
            ),
            WorkerLiteralDTO(value="a_b_c_d_e"),
            WorkerSMPCDTO(
                value=SMPCTablesInfo(
                    template=TableInfo(
                        name="a_b_c_d_e",
                        schema_=TableSchema(
                            columns=[ColumnInfo(name="test", dtype=DType.INT)]
                        ),
                        type_=TableType.NORMAL,
                    )
                )
            ),
        ],
    ]


@pytest.mark.parametrize("args", get_udf_args_cases())
def test_pos_udf_arguments_correct_resolutions(args):
    pos_args = WorkerUDFPosArguments(args=args)
    pos_args_json = pos_args.json()

    pos_args_unpacked = WorkerUDFPosArguments.parse_raw(pos_args_json)

    assert pos_args == pos_args_unpacked


@pytest.mark.parametrize("args", get_udf_args_cases())
def test_kw_udf_arguments_correct_resolutions(args):
    kw_args = WorkerUDFKeyArguments(args={pos: arg for pos, arg in enumerate(args)})
    kw_args_json = kw_args.json()

    kw_args_unpacked = WorkerUDFKeyArguments.parse_raw(kw_args_json)

    assert kw_args == kw_args_unpacked


def get_udf_results_cases():
    return [
        WorkerUDFResults(
            results=[
                WorkerTableDTO(
                    value=TableInfo(
                        name="a_b_c_d_e",
                        schema_=TableSchema(
                            columns=[ColumnInfo(name="test", dtype=DType.INT)]
                        ),
                        type_=TableType.NORMAL,
                    )
                ),
            ],
        ),
        WorkerUDFResults(
            results=[
                WorkerSMPCDTO(
                    value=SMPCTablesInfo(
                        template=TableInfo(
                            name="a_b_c_d_e",
                            schema_=TableSchema(
                                columns=[ColumnInfo(name="test", dtype=DType.INT)]
                            ),
                            type_=TableType.NORMAL,
                        )
                    )
                ),
            ],
        ),
        WorkerUDFResults(
            results=[
                WorkerTableDTO(
                    value=TableInfo(
                        name="a_b_c_d_e",
                        schema_=TableSchema(
                            columns=[ColumnInfo(name="test", dtype=DType.INT)]
                        ),
                        type_=TableType.NORMAL,
                    )
                ),
                WorkerSMPCDTO(
                    value=SMPCTablesInfo(
                        template=TableInfo(
                            name="a_b_c_d_e",
                            schema_=TableSchema(
                                columns=[ColumnInfo(name="test", dtype=DType.INT)]
                            ),
                            type_=TableType.NORMAL,
                        )
                    )
                ),
            ],
        ),
    ]


@pytest.mark.parametrize("udf_results", get_udf_results_cases())
def test_udf_results_correct_resolutions(udf_results):
    udf_results_json = udf_results.json()
    udf_results_unpacked = WorkerUDFResults.parse_raw(udf_results_json)
    assert udf_results == udf_results_unpacked
