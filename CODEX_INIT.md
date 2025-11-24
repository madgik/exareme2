# Codex Init — Exaflow Algorithms

Read this first whenever you open the repository through `/init`. It distills
how Exaflow’s native federated pipeline (ignoring Flower/SMPC add-ons) is
wired: where algorithms live, which configs they need, how requests travel from
the CLI to workers, and which commands are essential.

______________________________________________________________________

## Orientation

- **Purpose:** Exaflow is the execution engine behind the Medical Informatics
  Platform. It orchestrates *classical* algorithms across multiple
  workers that hold DuckDB-fed datasets.
- **Primary stack:** Python 3.10, Poetry, Quart + Hypercorn REST controller,
  gRPC worker services, DuckDB for local storage, optional aggregation server for
  vector aggregations.
- **Key dirs to inspect:**
  | Path | Why it matters |
  | --- | --- |
  | `exaflow/controller/quart` | HTTP endpoints; `endpoints.py` drives `/algorithms` (see `exaflow/controller/quart/endpoints.py:1`). |
  | `exaflow/controller/services/exaflow` | Controller-side strategy + worker task abstractions. |
  | `exaflow/algorithms/exaflow` | Algorithm implementations and JSON specs. |
  | `exaflow/worker` | gRPC server, DuckDB loader, UDF runner. |
  | `aggregation_server/` | Optional microservice providing SUM/MIN/MAX aggregation. |
  | `tasks.py` | `invoke` tasks for configs, data seeding, service lifecycle. |
  | `tests/algorithms` | Expected inputs/outputs used in validation. |

______________________________________________________________________

## Deep Dive: Exaflow Algorithm Lifecycle

The pipeline below is what you usually need to reference/debug.

1. **HTTP Request Intake**

   - `run_algorithm` (root script) posts to `POST /algorithms/<algorithm_name>`.
   - Quart wiring lives in `exaflow/controller/quart/endpoints.py:32`. It parses JSON
     into `AlgorithmRequestDTO`, validates it against enabled specs, and instantiates a
     strategy via `get_algorithm_execution_strategy`.

1. **Strategy Selection & Metadata Prep**

   - Exaflow algorithms use `ExaflowStrategy` / `ExaflowWithAggregationServerStrategy`
     (`exaflow/controller/services/exaflow/strategies.py:16`).
   - Strategy pulls metadata through the Worker Landscape Aggregator (datasets,
     variables, CDES) to validate that inputs exist on every worker.
   - If preprocessing requests longitudinal transforms, it calls
     `prepare_longitudinal_transformation` before algorithm execution.

1. **Algorithm Instantiation**

   - Algorithms are registered in `exaflow/exaflow_algorithm_classes` and implemented
     in `exaflow/algorithms/exaflow/*.py`. Each algorithm has a matching JSON spec
     describing required inputs/parameters (`*.json`).
   - Strategy creates the algorithm class, passing:
     - `Inputdata` payload (datasets, vars, parameters)
     - `ExaflowAlgorithmFlowEngineInterface` (see below)
     - Controller-side parameters

1. **Flow Engine & Worker Calls**

   - `ExaflowAlgorithmFlowEngineInterface` (`exaflow/controller/services/exaflow/algorithm_flow_engine_interface.py:1`)
     wraps parallel UDF dispatch. It:
     - Retrieves the correct worker UDF key through the `exaflow_registry`.
     - Injects preprocessing/raw-input metadata to each UDF call.
     - Runs requests concurrently via a thread pool, calling
       `ExaflowTasksHandler.run_udf` which in turn proxies to
       `WorkerTasksHandler` (gRPC client).
   - Each UDF is tagged with `@exaflow_udf` (see `exaflow/algorithms/exaflow/exaflow_registry.py:28`)
     which registers it and (optionally) flags whether aggregation server support is required.

1. **Worker Execution Path**

   - Worker gRPC server implementation is at `exaflow/worker/grpc_server.py`.
   - After startup it eagerly loads DuckDB datasets via
     `data_loader_service.load_data_folder`.
   - `WorkerTasksHandler` calls `RunUdf` on the worker; `udf_service` loads the
     registered UDF, applies parameters, and runs queries locally (usually through
     helpers in `exaflow/algorithms/exaflow/data_loading.py` and `library/`).
   - Results are JSON-serialisable dicts that the controller stitches into the
     final `AlgorithmResult` Pydantic model returned by `algorithm.run`.

1. **Aggregation Server (optional but part of Exaflow)**

   - Some UDFs set `with_aggregation_server=True`. `ExaflowWithAggregationServerStrategy`
     wraps execution with `ControllerAggregationClient` so the workers can push partial
     vectors to the gRPC aggregation service and retrieve the combined result.
   - Service config sits at `aggregation_server/config.toml`; start it with
     `inv start-aggregation-server`.

1. **Response**

   - Algorithm `.run(metadata)` returns a Pydantic model; the strategy serialises it
     to JSON and returns it as the HTTP response body.

______________________________________________________________________

## Working on Algorithms

- **Specs:** Algorithm metadata shipped to clients lives in `exaflow/algorithms/exaflow/*.json`.
  Update both spec and implementation when adding parameters or outputs.
- **Implementations:** `exaflow/algorithms/exaflow/*.py` typically define:
  - `ALGORITHM_SPEC` loading the JSON file.
  - A class derived from a base (e.g., `Algorithm` in `algorithm.py`) exposing `run`.
  - UDF helpers decorated with `@exaflow_udf`.
- **Data helpers:** `data_loading.py`, `metrics.py`, and `library/` hold reusable
  computations; prefer extending them before inlining SQL.
- **Controller integration:** Register new algorithms via
  `exaflow/exaflow_algorithm_classes` so the factory can find them.

______________________________________________________________________

## Testing & Validation

- **Pytest entrypoints:**
  - `poetry run pytest tests/algorithms` — golden tests for algorithm outputs.
- **Markers:** Defined in `pyproject.toml` (`slow`, `very_slow`, `smpc`, etc.).
  Focus on `slow/very_slow` when algorithm changes might affect distributed runs.
