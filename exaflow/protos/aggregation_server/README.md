## gRPC stub generation

From the repository root, regenerate the Python stubs with:

```bash
poetry run python -m grpc_tools.protoc \
  -I . \
  --python_out=. \
  --grpc_python_out=. \
  exaflow/protos/aggregation_server/aggregation_server.proto
```
