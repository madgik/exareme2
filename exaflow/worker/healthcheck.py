import grpc

from exaflow.worker import config as worker_config

from . import worker_pb2
from . import worker_pb2_grpc

target = f"{worker_config.grpc.ip}:{worker_config.grpc.port}"
channel = grpc.insecure_channel(target)
stub = worker_pb2_grpc.WorkerServiceStub(channel)

stub.Healthcheck(
    worker_pb2.HealthcheckRequest(request_id="HEALTHCHECK", check_db=True),
    timeout=worker_config.worker_tasks.tasks_timeout,
)

print("Healthcheck successful!")
