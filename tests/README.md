## Test Environments

There are 4 test folders each one running in different test environments.

### No Environment

The standalone tests are run:

- only with the basic infrastructure docker images pre-built,
- without an environment and
  are supposed to take care (create/cleanup) of any external dependency they need.

### Prod Environment

These tests are run:

- with all (worker, controller, aggregation_server) docker images pre-built,
- with 4 workers(globalworker, 3 localworkers), 1 controller and 1 aggregation server,
- with the production algorithms loaded,
- with data paths for each worker configured by the `worker_data_path_builder.py` script under `tests/test_data/.data_paths/<worker_name>`,
- using `helm` charts and `kind` for a pseudo federated kubernetes environment,
- having only the controller endpoints exposed and
- the tests contained can only be of production nature.

### SMPC Environment

These tests are run:

- with 3 workers(globalworker, 2 localworkers), 1 controller and 1 aggregation server,
- with data paths for each worker configured by the `worker_data_path_builder.py` script under `tests/test_data/.data_paths/<worker_name>`,
- with the production algorithms loaded,
- using `helm` charts and `kind` for a pseudo federated kubernetes environment,
- having only the controller endpoints exposed and
- the tests contained can only be of production nature.

### Algorithm Validation Tests

These tests are run:

- with 1 localworker, 1 globalworker, 1 controller and 1 aggregation server,
- with 5 localworkers, 1 globalworker, 1 controller and 1 aggregation server in a different job,
- with data paths for each worker configured by the `worker_data_path_builder.py` script under `tests/test_data/.data_paths/<worker_name>`,
- with the production algorithms loaded,
- using the `inv deploy` command of the `tasks.py` with a `.deployment.toml` template,
- can be run based on the workers' information in the `.deployment.toml`.
