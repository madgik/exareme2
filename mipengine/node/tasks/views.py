import json

from celery import shared_task

from mipengine.node.monetdb_interface import views
from mipengine.node.monetdb_interface.common import config
from mipengine.node.monetdb_interface.common import create_table_name
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
        str(TableSchema)
            A TableSchema object in a jsonified format
    """
    schema = views.get_view_schema(view_name)
    return schema.to_json()


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
def create_view(context_id: str, command_id: str, pathology: str, datasets_json: str, columns_json: str, filters_json: str) -> str:
    #  filter: str, x: Optional[List[str]], y: Optional[List[str]]
    # We need to refactor that
    # pathology and filter will not be used for now, but should exist on the interface
    """
        Parameters
        ----------
        context_id : str
            The id of the experiment
        command_id : str
            The id of the command that the view
        pathology : str
            The pathology data table on which the view will be created
        datasets_json : str(List[str])
            A list of dataset names in a jsonified format
        columns_json : str(List[str])
            A list of column names in a jsonified format
        filters_json : str(dict)
            A Jquery filters object

        Returns
        ------
        str
            The name of the created view in lower case
    """
    view_name = create_table_name("view", command_id, context_id, config["node"]["identifier"])
    views.create_view(view_name, pathology, json.loads(datasets_json), json.loads(columns_json), filters_json)
    return view_name.lower()
