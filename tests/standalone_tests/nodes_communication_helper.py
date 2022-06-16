def get_celery_task_signature(celery_app, task):
    signature_mapping = {
        "get_node_info": celery_app.signature(
            "mipengine.node.tasks.common.get_node_info"
        ),
        "create_table": celery_app.signature(
            "mipengine.node.tasks.tables.create_table"
        ),
        "get_tables": celery_app.signature("mipengine.node.tasks.tables.get_tables"),
        "insert_data_to_table": celery_app.signature(
            "mipengine.node.tasks.tables.insert_data_to_table"
        ),
        "create_view": celery_app.signature("mipengine.node.tasks.views.create_view"),
        "create_data_model_views": celery_app.signature(
            "mipengine.node.tasks.views.create_data_model_views"
        ),
        "get_views": celery_app.signature("mipengine.node.tasks.views.get_views"),
        "create_merge_table": celery_app.signature(
            "mipengine.node.tasks.merge_tables.create_merge_table"
        ),
        "get_merge_tables": celery_app.signature(
            "mipengine.node.tasks.merge_tables.get_merge_tables"
        ),
        "create_remote_table": celery_app.signature(
            "mipengine.node.tasks.remote_tables.create_remote_table"
        ),
        "get_remote_tables": celery_app.signature(
            "mipengine.node.tasks.remote_tables.get_remote_tables"
        ),
        "get_table_schema": celery_app.signature(
            "mipengine.node.tasks.common.get_table_schema"
        ),
        "get_table_data": celery_app.signature(
            "mipengine.node.tasks.common.get_table_data"
        ),
        "get_udf": celery_app.signature("mipengine.node.tasks.udfs.get_udf"),
        "run_udf": celery_app.signature("mipengine.node.tasks.udfs.run_udf"),
        "get_run_udf_query": celery_app.signature(
            "mipengine.node.tasks.udfs.get_run_udf_query"
        ),
        "clean_up": celery_app.signature("mipengine.node.tasks.common.clean_up"),
        "validate_smpc_templates_match": celery_app.signature(
            "mipengine.node.tasks.smpc.validate_smpc_templates_match"
        ),
        "load_data_to_smpc_client": celery_app.signature(
            "mipengine.node.tasks.smpc.load_data_to_smpc_client"
        ),
        "get_smpc_result": celery_app.signature(
            "mipengine.node.tasks.smpc.get_smpc_result"
        ),
    }

    if task not in signature_mapping.keys():
        raise ValueError(f"Task: {task} is not a valid task.")
    return signature_mapping.get(task)
