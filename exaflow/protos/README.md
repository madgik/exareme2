## gRPC stub generation

From the repository root, regenerate the Python stubs with:

Aggregation server:

```bash
poetry run python -m grpc_tools.protoc \
  -I . \
  --python_out=. \
  --grpc_python_out=. \
  exaflow/protos/aggregation_server/aggregation_server.proto
```

Worker service:

```bash
poetry run python -m grpc_tools.protoc \
  -I . \
  --python_out=. \
  --grpc_python_out=. \
  exaflow/protos/worker/worker.proto
```
