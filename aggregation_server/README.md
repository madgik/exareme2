# Exaflow Aggregation Server

A lightweight gRPC microservice that performs **federated vector aggregation**
(`SUM`, `MIN`, `MAX`) from multiple worker nodes.

______________________________________________________________________

## Features

| Capability | Description |
| --- | --- |
| **gRPC API** | Defined in [`aggregation_server.proto`](aggregation_server.proto) – unary RPCs `Configure`, `Aggregate`, `Cleanup`, `Unregister`. |
| **Aggregation modes** | `SUM`, `MIN`, `MAX` over vectors; `Aggregate` accepts repeated operations in one call. |
| **Payload formats** | Vectors can be sent as repeated doubles or Arrow tensor bytes. |
| **Concurrency** | Thread-pool server (configurable worker pool). |
| **Request lifecycle** | A `request_id` can span multiple steps until `Cleanup` resets the context. |

## Docker

Build a production-ready image with the supplied **Dockerfile**:

```bash
docker build -t exaflow/aggregation_server:latest .
docker run -p 50051:50051 exaflow/aggregation_server:latest
```
