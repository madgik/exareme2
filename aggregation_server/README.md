# Exareme-2 Aggregation Server

A lightweight gRPC microservice that performs **federated vector aggregation**
(`SUM`, `MIN`, `MAX`) from multiple worker nodes.

______________________________________________________________________

## Features

## | Capability   | Description | |--------------|-------------| | **gRPC API** | Defined in [`aggregation_server.proto`](aggregation_server.proto) – three unary RPCs (`Configure`, `Aggregate`, `Cleanup`). | | **Concurrency** | Thread-pool server (configurable worker pool). | | **Idempotent requests** | Same `request_id` may be re-used for multiple aggregation rounds. |

## Docker

Build a production-ready image with the supplied **Dockerfile**:

```bash
docker build -t exareme2/aggregation_server:latest .
docker run -p 50051:50051 exareme2/aggregation_server:latest
```
