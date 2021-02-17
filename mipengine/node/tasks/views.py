import json

from celery import shared_task

from mipengine.node.monetdb_interface import views
from mipengine.node.monetdb_interface.common import config
from mipengine.node.monetdb_interface.common import create_table_name
from mipengine.node.tasks.data_classes import ColumnInfo
from mipengine.node.tasks.data_classes import TableData


@shared_task
def get_views(context_id: str) -> str:
    """
        Parameters
        ----------
        context_id : str
            The id of the experiment

        Returns
        ------
        str
            A list of view names in a jsonified format
    """
    return json.dumps(views.get_views_names(context_id))


@shared_task
def get_view_schema(view_name: str) -> str:
    """
        Parameters
        ----------
        view_name : str
            The name of the view

        Returns
        ------
        A schema(list of ColumnInfo's objects) in a jsonified format
    """
    schema = views.get_view_schema(view_name)
    return ColumnInfo.schema().dumps(schema, many=True)


@shared_task
def get_view_data(view_name: str) -> str:
    """
        Parameters
        ----------
        view_name : str
        The name of the view

        Returns
        ------
        str
            An object of TableData in a jsonified format
    """
    schema = views.get_view_schema(view_name)
    data = views.get_view_data(view_name)
    return TableData(schema, data).to_json()


@shared_task
def create_view(context_id: str, columns_json: str, datasets_json: str) -> str:
    # TODO The parameters should be context_id, pathology:str, datasets:List[str],
    #  filter: str, x: Optional[List[str]], y: Optional[List[str]]
    # We need to refactor that
    # pathology and filter will not be used for now, but should exist on the interface
    """
        Parameters
        ----------
        context_id : str
            The id of the experiment
        columns_json : str
            A list of column names in a jsonified format
        datasets_json : str
            A list of dataset names in a jsonified format

        Returns
        ------
        str
            The name of the created view in lower case
    """
    view_name = create_table_name("view", context_id, config["node"]["identifier"])
    views.create_view(view_name, json.loads(columns_json), json.loads(datasets_json))
    return view_name.lower()
