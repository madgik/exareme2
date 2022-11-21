from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from mipengine import DType as dt
from mipengine.udfgen.ast import ASTNode
from mipengine.udfgen.ast import Imports
from mipengine.udfgen.ast import LiteralAssignments
from mipengine.udfgen.ast import LoggerAssignment
from mipengine.udfgen.ast import PlaceholderAssignments
from mipengine.udfgen.ast import TableBuilds
from mipengine.udfgen.ast import UDFBody
from mipengine.udfgen.ast import UDFBodyStatements
from mipengine.udfgen.ast import UDFLoopbackReturnStatements
from mipengine.udfgen.ast import UDFReturnStatement
from mipengine.udfgen.helpers import is_any_element_of_type
from mipengine.udfgen.iotypes import MAIN_TABLE_PLACEHOLDER
from mipengine.udfgen.iotypes import DictArg
from mipengine.udfgen.iotypes import DictType
from mipengine.udfgen.iotypes import InputType
from mipengine.udfgen.iotypes import LiteralArg
from mipengine.udfgen.iotypes import LoopbackOutputType
from mipengine.udfgen.iotypes import MergeTransferType
from mipengine.udfgen.iotypes import OutputType
from mipengine.udfgen.iotypes import PlaceholderArg
from mipengine.udfgen.iotypes import StateType
from mipengine.udfgen.iotypes import TableArg
from mipengine.udfgen.iotypes import TransferType
from mipengine.udfgen.iotypes import TransferTypeBase
from mipengine.udfgen.iotypes import UDFArgument
from mipengine.udfgen.iotypes import UDFLoggerArg

LN = "\n"


# ~~~~~~ IOTypes ~~~~~~ #


class SecureTransferType(DictType, InputType, LoopbackOutputType):
    _data_column_name = "secure_transfer"
    _data_column_type = dt.JSON
    _sum_op: bool
    _min_op: bool
    _max_op: bool

    def __init__(self, sum_op=False, min_op=False, max_op=False):
        self._sum_op = sum_op
        self._min_op = min_op
        self._max_op = max_op

    @property
    def sum_op(self):
        return self._sum_op

    @property
    def min_op(self):
        return self._min_op

    @property
    def max_op(self):
        return self._max_op

    def get_build_template(self):
        colname = self.data_column_name
        return LN.join(
            [
                f'__transfer_strs = _conn.execute("SELECT {colname} from {{table_name}};")["{colname}"]',
                "__transfers = [json.loads(str) for str in __transfer_strs]",
                "{varname} = udfio.secure_transfers_to_merged_dict(__transfers)",
            ]
        )

    def get_main_return_stmt_template(self) -> str:
        return "return json.dumps({return_name})"

    def get_secondary_return_stmt_template(self, tablename_placeholder) -> str:
        return (
            '_conn.execute(f"INSERT INTO $'
            + tablename_placeholder
            + " VALUES ('{{json.dumps({return_name})}}');\")"
        )


TransferTypeBase.register(SecureTransferType)


class SMPCSecureTransferType(SecureTransferType):
    def get_build_template(self):
        colname = self.data_column_name
        return LN.join(
            [
                f'__transfer_strs = _conn.execute("SELECT {colname} FROM {{table_name}};")["{colname}"]',
                "__transfers = [json.loads(str) for str in __transfer_strs]",
                "{varname} = udfio.secure_transfers_to_merged_dict(__transfers)",
            ]
        )

    def get_main_return_stmt_template(self) -> str:
        return_stmts = [
            "template, sum_op, min_op, max_op = udfio.split_secure_transfer_dict({return_name})"
        ]
        (
            _,
            sum_op_tmpl,
            min_op_tmpl,
            max_op_tmpl,
        ) = get_smpc_table_template_names(MAIN_TABLE_PLACEHOLDER)
        return_stmts.extend(
            self._get_secure_transfer_op_return_stmt_template(
                self.sum_op, sum_op_tmpl, "sum_op"
            )
        )
        return_stmts.extend(
            self._get_secure_transfer_op_return_stmt_template(
                self.min_op, min_op_tmpl, "min_op"
            )
        )
        return_stmts.extend(
            self._get_secure_transfer_op_return_stmt_template(
                self.max_op, max_op_tmpl, "max_op"
            )
        )
        return_stmts.append("return json.dumps(template)")
        return LN.join(return_stmts)

    def get_secondary_return_stmt_template(self, tablename_placeholder) -> str:
        return_stmts = [
            "template, sum_op, min_op, max_op = udfio.split_secure_transfer_dict({return_name})"
        ]
        (
            template_tmpl,
            sum_op_tmpl,
            min_op_tmpl,
            max_op_tmpl,
        ) = get_smpc_table_template_names(tablename_placeholder)
        return_stmts.append(
            '_conn.execute(f"INSERT INTO $'
            + template_tmpl
            + " VALUES ('{{json.dumps(template)}}');\")"
        )
        return_stmts.extend(
            self._get_secure_transfer_op_return_stmt_template(
                self.sum_op, sum_op_tmpl, "sum_op"
            )
        )
        return_stmts.extend(
            self._get_secure_transfer_op_return_stmt_template(
                self.min_op, min_op_tmpl, "min_op"
            )
        )
        return_stmts.extend(
            self._get_secure_transfer_op_return_stmt_template(
                self.max_op, max_op_tmpl, "max_op"
            )
        )
        return LN.join(return_stmts)

    @staticmethod
    def _get_secure_transfer_op_return_stmt_template(
        op_enabled, table_name_tmpl, op_name
    ):
        if not op_enabled:
            return []
        return [
            '_conn.execute(f"INSERT INTO $'
            + table_name_tmpl
            + f" VALUES ('{{{{json.dumps({op_name})}}}}');\")"
        ]

    def get_smpc_build_template(self):
        def get_smpc_op_template(enabled, operation_name):
            stmts = []
            if enabled:
                stmts.append(
                    f'__{operation_name}_values_str = _conn.execute("SELECT secure_transfer from {{{operation_name}_values_table_name}};")["secure_transfer"][0]'
                )
                stmts.append(
                    f"__{operation_name}_values = json.loads(__{operation_name}_values_str)"
                )
            else:
                stmts.append(f"__{operation_name}_values = None")
            return stmts

        stmts = []
        stmts.append(
            '__template_str = _conn.execute("SELECT secure_transfer from {template_table_name};")["secure_transfer"][0]'
        )
        stmts.append("__template = json.loads(__template_str)")
        stmts.extend(get_smpc_op_template(self.sum_op, "sum_op"))
        stmts.extend(get_smpc_op_template(self.min_op, "min_op"))
        stmts.extend(get_smpc_op_template(self.max_op, "max_op"))
        stmts.append(
            "{varname} = udfio.construct_secure_transfer_dict(__template,__sum_op_values,__min_op_values,__max_op_values)"
        )
        return LN.join(stmts)

    @classmethod
    def cast(cls, obj):
        obj.__class__ = cls
        return obj


def get_smpc_table_template_names(prefix: str):
    """
    This is used when a secure transfer is returned with smpc enabled.
    The secure_transfer is one output_type but needs to be broken into
    multiple tables, hence more than one main table names are needed.
    """
    return (
        prefix,
        prefix + "_sum_op",
        prefix + "_min_op",
        prefix + "_max_op",
    )


def secure_transfer(sum_op=False, min_op=False, max_op=False):
    if not sum_op and not min_op and not max_op:
        raise ValueError(
            "In a secure_transfer at least one operation should be enabled."
        )
    return SecureTransferType(sum_op, min_op, max_op)


class SecureTransferArg(DictArg):
    type = SecureTransferType()

    def __init__(self, table_name: str):
        super().__init__(table_name)


class SMPCSecureTransferArg(UDFArgument):
    type: SecureTransferType
    template_table_name: str
    sum_op_values_table_name: str
    min_op_values_table_name: str
    max_op_values_table_name: str

    def __init__(
        self,
        template_table_name: str,
        sum_op_values_table_name: str,
        min_op_values_table_name: str,
        max_op_values_table_name: str,
    ):
        sum_op = False
        min_op = False
        max_op = False
        if sum_op_values_table_name:
            sum_op = True
        if min_op_values_table_name:
            min_op = True
        if max_op_values_table_name:
            max_op = True
        self.type = SMPCSecureTransferType(sum_op, min_op, max_op)
        self.template_table_name = template_table_name
        self.sum_op_values_table_name = sum_op_values_table_name
        self.min_op_values_table_name = min_op_values_table_name
        self.max_op_values_table_name = max_op_values_table_name


# ~~~~~~ AST ~~~~~~ #


class UDFBodySMPC(UDFBody):
    def __init__(
        self,
        table_args: Dict[str, TableArg],
        smpc_args: Dict[str, SMPCSecureTransferArg],
        literal_args: Dict[str, LiteralArg],
        logger_arg: Optional[Tuple[str, UDFLoggerArg]],
        placeholder_args: Dict[str, PlaceholderArg],
        statements: list,
        main_return_name: str,
        main_return_type: OutputType,
        sec_return_names: List[str],
        sec_return_types: List[OutputType],
    ):
        all_types = (
            [arg.type for arg in table_args.values()]
            + [main_return_type]
            + sec_return_types
        )

        import_pickle = is_any_element_of_type(StateType, all_types)
        import_json = is_any_element_of_type(
            (TransferType, SecureTransferType, MergeTransferType), all_types
        )

        self.statements = []

        # imports
        self.statements.append(
            Imports(
                import_pickle=import_pickle,
                import_json=import_json,
            )
        )

        # initial assignments
        self.statements.append(TableBuilds(table_args))
        self.statements.append(SMPCBuilds(smpc_args))
        self.statements.append(LiteralAssignments(literal_args))
        self.statements.append(LoggerAssignment(logger_arg))
        self.statements.append(PlaceholderAssignments(placeholder_args))

        # main body
        self.statements.append(UDFBodyStatements(statements))

        # return statements
        self.statements.append(
            UDFLoopbackReturnStatements(
                sec_return_names=sec_return_names,
                sec_return_types=sec_return_types,
            )
        )
        self.statements.append(UDFReturnStatement(main_return_name, main_return_type))


class SMPCBuild(ASTNode):
    def __init__(self, arg_name, arg, template):
        self.arg_name = arg_name
        self.arg = arg
        self.template = template

    def compile(self) -> str:
        return self.template.format(
            varname=self.arg_name,
            template_table_name=self.arg.template_table_name,
            sum_op_values_table_name=self.arg.sum_op_values_table_name,
            min_op_values_table_name=self.arg.min_op_values_table_name,
            max_op_values_table_name=self.arg.max_op_values_table_name,
        )


class SMPCBuilds(ASTNode):
    def __init__(self, smpc_args: Dict[str, SMPCSecureTransferArg]):
        self.smpc_builds = [
            SMPCBuild(arg_name, arg, template=arg.type.get_smpc_build_template())
            for arg_name, arg in smpc_args.items()
        ]

    def compile(self) -> str:
        return LN.join([tb.compile() for tb in self.smpc_builds])
