signature_mapping = {
    "get_node_info": "exareme2.node.tasks.common.get_node_info",
    "get_data_model_cdes": "exareme2.node.tasks.common.get_data_model_cdes",
    "get_node_datasets_per_data_model": "exareme2.node.tasks.common.get_node_datasets_per_data_model",
    "get_data_model_attributes": "exareme2.node.tasks.common.get_data_model_attributes",
    "create_table": "exareme2.node.tasks.tables.create_table",
    "get_tables": "exareme2.node.tasks.tables.get_tables",
    "insert_data_to_table": "exareme2.node.tasks.tables.insert_data_to_table",
    "create_view": "exareme2.node.tasks.views.create_view",
    "create_data_model_views": "exareme2.node.tasks.views.create_data_model_views",
    "get_views": "exareme2.node.tasks.views.get_views",
    "create_merge_table": "exareme2.node.tasks.merge_tables.create_merge_table",
    "get_merge_tables": "exareme2.node.tasks.merge_tables.get_merge_tables",
    "create_remote_table": "exareme2.node.tasks.remote_tables.create_remote_table",
    "get_remote_tables": "exareme2.node.tasks.remote_tables.get_remote_tables",
    "get_table_data": "exareme2.node.tasks.common.get_table_data",
    "get_udf": "exareme2.node.tasks.udfs.get_udf",
    "run_udf": "exareme2.node.tasks.udfs.run_udf",
    "get_run_udf_query": "exareme2.node.tasks.udfs.get_run_udf_query",
    "cleanup": "exareme2.node.tasks.common.cleanup",
    "validate_smpc_templates_match": "exareme2.node.tasks.smpc.validate_smpc_templates_match",
    "load_data_to_smpc_client": "exareme2.node.tasks.smpc.load_data_to_smpc_client",
    "get_smpc_result": "exareme2.node.tasks.smpc.get_smpc_result",
}


def get_celery_task_signature(task):
    if task not in signature_mapping.keys():
        raise ValueError(f"Task: {task} is not a valid task.")
    return signature_mapping[task]
