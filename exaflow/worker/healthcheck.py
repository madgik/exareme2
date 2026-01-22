import grpc
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc

from exaflow.worker import config as worker_config

from . import worker_pb2
from . import worker_pb2_grpc

target = f"{worker_config.grpc.ip}:{worker_config.grpc.port}"
channel = grpc.insecure_channel(target)

health_stub = health_pb2_grpc.HealthStub(channel)
health_response = health_stub.Check(
    health_pb2.HealthCheckRequest(service="worker"),
    timeout=worker_config.worker_tasks.tasks_timeout,
)
if health_response.status != health_pb2.HealthCheckResponse.SERVING:
    raise RuntimeError(
        f"Worker health status is {health_response.status}, expected SERVING."
    )

worker_stub = worker_pb2_grpc.WorkerServiceStub(channel)
worker_stub.Healthcheck(
    worker_pb2.HealthcheckRequest(request_id="HEALTHCHECK", check_db=True),
    timeout=worker_config.worker_tasks.tasks_timeout,
)

print("Healthcheck successful!")
