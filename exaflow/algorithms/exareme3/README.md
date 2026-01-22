# Exaflow Algorithms – Author Guide

This folder contains the exaflow algorithm flows migrated from exareme2. This guide highlights the key pieces you need to know when adding or modifying a flow.

## Core contracts

- Base class: `algorithm.py.Algorithm`
  - `algname` must match the `"name"` in the `<algorithm>.json`.
  - `engine` exposes `run_algorithm_udf` to execute UDFs registered with `exareme3_registry.exaflow_udf`.
  - `inputdata` is a pydantic model with `x`, `y`, `datasets`, `filters` (see `exaflow/algorithms/utils/inputdata_utils.Inputdata`). Algorithms should validate required vars via `validation_utils`.
  - `metadata` passed to `run()` is a `dict[var] -> {is_categorical: bool, enumerations: {...}}`. Use `metadata_utils.validate_metadata_vars` / `validate_metadata_enumerations` to enforce presence.
  - Parameters: `parameters` is a plain dict from the JSON spec.

## UDFs and aggregation

- Decorate UDFs with `@exaflow_udf(...)` (see `exareme3_registry.py`).
  - `with_aggregation_server=True` injects `agg_client`, which implements sum/min/max over numpy-like arrays (`ExaflowUDFAggregationClientI`).
  - UDF registry keys are derived from `__qualname__` and module; duplicates raise to avoid ambiguity.
- Lazy aggregation helper: `library/lazy_aggregation.lazy_agg` can batch `agg_client.sum/min/max` calls when supported.

## Preprocessing and metadata helpers

- Metadata validation: `metadata_utils.validate_metadata_vars` (requires `is_categorical`), `validate_metadata_enumerations` (requires `enumerations`).
- Variable checks: `validation_utils` has `require_dependent_var`, `require_covariates`, and exact-count variants.
- Dummy encoding: use `preprocessing.get_dummy_categories` with a collect UDF to discover categories, then `metrics.build_design_matrix` inside UDFs to build the matrix.

## Cross-validation utilities

- `crossvalidation.kfold_indices` yields train/test index arrays for K-fold splits without sklearn.
- `crossvalidation.split_dataframe` yields (train_df, test_df) pairs for pandas DataFrames.

## Patterns to follow

- Always validate input variables and metadata at the start of `run()`.
- Keep UDF inputs minimal (only the columns you use) to reduce network/serialization overhead.
- When using aggregation server, aggregate numpy arrays and convert to lists before returning to stay JSON-serializable.
- Preserve privacy checks (e.g., minimum row count masking) when aggregating counts/histograms.
