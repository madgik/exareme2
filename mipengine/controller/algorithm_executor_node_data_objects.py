from abc import ABC


class NodeData(ABC):
    """
    NodeData is an object representing data located into one specific Node.
    """

    pass


class TableName(NodeData):
    def __init__(self, table_name):
        self._full_name = table_name
        full_name_split = self._full_name.split("_")
        self._table_type = full_name_split[0]
        self._node_id = full_name_split[1]
        self._context_id = full_name_split[2]
        self._command_id = full_name_split[3]
        self._command_subid = full_name_split[4]

    @property
    def full_table_name(self) -> str:
        return self._full_name

    @property
    def table_type(self) -> str:
        return self._table_type

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def context_id(self) -> str:
        return self._context_id

    @property
    def command_id(self) -> str:
        return self._command_id

    @property
    def command_subid(self) -> str:
        return self._command_subid

    def without_node_id(self) -> str:
        return (
            self._table_type
            + "_"
            + self._context_id
            + "_"
            + self._command_id
            + "_"
            + self._command_subid
        )

    def __repr__(self):
        return self.full_table_name


class SMPCTableNames(NodeData):
    template: TableName
    sum_op: TableName
    min_op: TableName
    max_op: TableName
    union_op: TableName

    def __init__(self, template, sum_op, min_op, max_op, union_op):
        self.template = template
        self.sum_op = sum_op
        self.min_op = min_op
        self.max_op = max_op
        self.union_op = union_op
