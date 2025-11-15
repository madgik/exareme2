signature_mapping = {
    "get_worker_info": "exareme2.worker.worker_info.worker_info_api.get_worker_info",
    "get_data_model_cdes": "exareme2.worker.worker_info.worker_info_api.get_data_model_cdes",
    "get_worker_datasets_per_data_model": "exareme2.worker.worker_info.worker_info_api.get_worker_datasets_per_data_model",
    "get_data_model_attributes": "exareme2.worker.worker_info.worker_info_api.get_data_model_attributes",
    "healthcheck": "exareme2.worker.worker_info.worker_info_api.healthcheck",
    "start_flower_client": "exareme2.worker.flower.starter.starter_api.start_flower_client",
    "start_flower_server": "exareme2.worker.flower.starter.starter_api.start_flower_server",
    "stop_flower_server": "exareme2.worker.flower.cleanup.cleanup_api.stop_flower_server",
    "stop_flower_client": "exareme2.worker.flower.cleanup.cleanup_api.stop_flower_client",
    "garbage_collect": "exareme2.worker.flower.cleanup.cleanup_api.garbage_collect",
}


def get_celery_task_signature(task):
    if task not in signature_mapping.keys():
        raise ValueError(f"Task: {task} is not a valid task.")
    return signature_mapping[task]
