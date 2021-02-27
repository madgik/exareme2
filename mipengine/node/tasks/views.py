import json
from typing import List

from celery import shared_task

from mipengine.node.monetdb_interface import views
from mipengine.node.monetdb_interface.common import config
from mipengine.node.monetdb_interface.common import create_table_name


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
    return views.get_views_names(context_id)


@shared_task
def create_view(context_id: str, command_id: str, pathology: str, datasets: List[str], columns: List[str], filters_json: str) -> str:
    # filter: str, x: Optional[List[str]], y: Optional[List[str]]
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
    view_name = create_table_name("view", command_id, context_id, config["node"]["identifier"])
    views.create_view(view_name=view_name,
                      pathology=pathology,
                      datasets=json.loads(datasets),
                      columns=json.loads(columns))
    return view_name.lower()
