signature_mapping = {
    "get_node_info": "mipengine.node.tasks.common.get_node_info",
    "create_table": "mipengine.node.tasks.tables.create_table",
    "get_tables": "mipengine.node.tasks.tables.get_tables",
    "insert_data_to_table": "mipengine.node.tasks.tables.insert_data_to_table",
    "create_view": "mipengine.node.tasks.views.create_view",
    "create_data_model_view": "mipengine.node.tasks.views.create_data_model_view",
    "get_views": "mipengine.node.tasks.views.get_views",
    "create_merge_table": "mipengine.node.tasks.merge_tables.create_merge_table",
    "get_merge_tables": "mipengine.node.tasks.merge_tables.get_merge_tables",
    "create_remote_table": "mipengine.node.tasks.remote_tables.create_remote_table",
    "get_remote_tables": "mipengine.node.tasks.remote_tables.get_remote_tables",
    "get_table_schema": "mipengine.node.tasks.common.get_table_schema",
    "get_table_data": "mipengine.node.tasks.common.get_table_data",
    "get_udf": "mipengine.node.tasks.udfs.get_udf",
    "run_udf": "mipengine.node.tasks.udfs.run_udf",
    "get_run_udf_query": "mipengine.node.tasks.udfs.get_run_udf_query",
    "clean_up": "mipengine.node.tasks.common.clean_up",
    "validate_smpc_templates_match": "mipengine.node.tasks.smpc.validate_smpc_templates_match",
    "load_data_to_smpc_client": "mipengine.node.tasks.smpc.load_data_to_smpc_client",
    "get_smpc_result": "mipengine.node.tasks.smpc.get_smpc_result",
}


def get_celery_task_signature(task):
    if task not in signature_mapping.keys():
        raise ValueError(f"Task: {task} is not a valid task.")
    return signature_mapping[task]
