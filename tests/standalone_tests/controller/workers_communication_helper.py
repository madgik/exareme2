signature_mapping = {
    "get_worker_info": "exareme2.worker.worker_info.worker_info_api.get_worker_info",
    "get_data_model_cdes": "exareme2.worker.worker_info.worker_info_api.get_data_model_cdes",
    "get_worker_datasets_per_data_model": "exareme2.worker.worker_info.worker_info_api.get_worker_datasets_per_data_model",
    "get_data_model_attributes": "exareme2.worker.worker_info.worker_info_api.get_data_model_attributes",
    "healthcheck": "exareme2.worker.worker_info.worker_info_api.healthcheck",
    "get_views": "exareme2.worker.exareme2.views.views_api.get_views",
    "create_view": "exareme2.worker.exareme2.views.views_api.create_view",
    "create_data_model_views": "exareme2.worker.exareme2.views.views_api.create_data_model_views",
    "create_table": "exareme2.worker.exareme2.tables.tables_api.create_table",
    "create_merge_table": "exareme2.worker.exareme2.tables.tables_api.create_merge_table",
    "create_remote_table": "exareme2.worker.exareme2.tables.tables_api.create_remote_table",
    "get_tables": "exareme2.worker.exareme2.tables.tables_api.get_tables",
    "get_merge_tables": "exareme2.worker.exareme2.tables.tables_api.get_merge_tables",
    "get_remote_tables": "exareme2.worker.exareme2.tables.tables_api.get_remote_tables",
    "get_table_data": "exareme2.worker.exareme2.tables.tables_api.get_table_data",
    "insert_data_to_table": "exareme2.worker.exareme2.tables.tables_api.insert_data_to_table",
    "run_udf": "exareme2.worker.exareme2.udfs.udfs_api.run_udf",
    "cleanup": "exareme2.worker.exareme2.cleanup.cleanup_api.cleanup",
    "validate_smpc_templates_match": "exareme2.worker.exareme2.smpc.smpc_api.validate_smpc_templates_match",
    "load_data_to_smpc_client": "exareme2.worker.exareme2.smpc.smpc_api.load_data_to_smpc_client",
    "get_smpc_result": "exareme2.worker.exareme2.smpc.smpc_api.get_smpc_result",
    "start_flower_client": "exareme2.worker.flower.starter.flower_api.start_flower_client",
    "start_flower_server": "exareme2.worker.flower.starter.flower_api.start_flower_server",
    "stop_flower_server": "exareme2.worker.flower.cleanup.cleanup_api.stop_flower_server",
    "stop_flower_client": "exareme2.worker.flower.cleanup.cleanup_api.stop_flower_client",
    "garbage_collect": "exareme2.worker.flower.cleanup.cleanup_api.garbage_collect",
}


def get_celery_task_signature(task):
    if task not in signature_mapping.keys():
        raise ValueError(f"Task: {task} is not a valid task.")
    return signature_mapping[task]
