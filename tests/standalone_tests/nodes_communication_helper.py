signature_mapping = {
    "get_node_info": "exareme2.node.celery.node_info.get_node_info",
    "get_data_model_cdes": "exareme2.node.celery.node_info.get_data_model_cdes",
    "get_node_datasets_per_data_model": "exareme2.node.celery.node_info.get_node_datasets_per_data_model",
    "get_data_model_attributes": "exareme2.node.celery.node_info.get_data_model_attributes",
    "healthcheck": "exareme2.node.celery.node_info.healthcheck",
    "get_views": "exareme2.node.celery.views.get_views",
    "create_view": "exareme2.node.celery.views.create_view",
    "create_data_model_views": "exareme2.node.celery.views.create_data_model_views",
    "create_table": "exareme2.node.celery.tables.create_table",
    "create_merge_table": "exareme2.node.celery.tables.create_merge_table",
    "create_remote_table": "exareme2.node.celery.tables.create_remote_table",
    "get_tables": "exareme2.node.celery.tables.get_tables",
    "get_merge_tables": "exareme2.node.celery.tables.get_merge_tables",
    "get_remote_tables": "exareme2.node.celery.tables.get_remote_tables",
    "get_table_data": "exareme2.node.celery.tables.get_table_data",
    "insert_data_to_table": "exareme2.node.celery.tables.insert_data_to_table",
    "run_udf": "exareme2.node.celery.udfs.run_udf",
    "cleanup": "exareme2.node.celery.cleanup.cleanup",
    "validate_smpc_templates_match": "exareme2.node.celery.smpc.validate_smpc_templates_match",
    "load_data_to_smpc_client": "exareme2.node.celery.smpc.load_data_to_smpc_client",
    "get_smpc_result": "exareme2.node.celery.smpc.get_smpc_result",
}


def get_celery_task_signature(task):
    if task not in signature_mapping.keys():
        raise ValueError(f"Task: {task} is not a valid task.")
    return signature_mapping[task]
