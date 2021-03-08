from typing import List

from celery import shared_task

from mipengine.node.monetdb_interface import views
from mipengine.node.monetdb_interface.common_actions import config
from mipengine.node.monetdb_interface.common_actions import create_table_name
from mipengine.node.monetdb_interface.connection_pool import get_connection, release_connection


@shared_task
def get_views(context_id: str) -> List[str]:
    """
        Parameters
        ----------
        context_id : str
            The id of the experiment

        Returns
        ------
        List[str]
            A list of view names
    """
    connection = get_connection()
    view_names = views.get_views_names(connection.cursor(), context_id)
    release_connection(connection)
    return view_names


@shared_task
def create_view(context_id: str,
                command_id: str,
                pathology: str,
                datasets: List[str],
                columns: List[str],
                filters_json: str
                ) -> str:
    # TODO We need to add the filters
    """
        Parameters
        ----------
        context_id : str
            The id of the experiment
        command_id : str
            The id of the command that the view
        pathology : str
            The pathology data table on which the view will be created
        datasets : List[str]
            A list of dataset names
        columns : List[str]
            A list of column names
        filters_json : str(dict)
            A Jquery filters object

        Returns
        ------
        str
            The name of the created view in lower case
    """
    connection = get_connection()
    view_name = create_table_name("view", command_id, context_id, config["node"]["identifier"])
    views.create_view(
        cursor=connection.cursor(),
        view_name=view_name,
        pathology=pathology,
        datasets=datasets,
        columns=columns)
    connection.commit()
    release_connection(connection)
    return view_name.lower()
