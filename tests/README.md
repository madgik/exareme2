## Test Environments

There are 4 test folders each one running in different test environments.

### No Environment

The standalone tests are run:

- only with the basic (monetdb) docker images pre-built,
- without an environment and
  are supposed to take care (create/cleanup) of any external dependency they need.

### Dev Environment (Deprecated)

These tests are run:

- with the basic (monetdb, rabbitmq) docker images pre-built,
- with 3 workers(globalworker, 2 localworkers) and 1 controller,
- with the test data loaded using `inv load-data` and
- with the production and testing algorithms loaded,
- using the `inv deploy` command of the `tasks.py` with a `.deployment.toml` template,
- can be run based on the workers' information in the `.deployment.toml`.

### Prod Environment

These tests are run:

- with all (monetdb, rabbitmq, worker, controller) docker images pre-built,
- with 4 workers(globalworker, 3 localworkers) and 1 controller,
- with the test data loaded through the mip_db container,
- with the production algorithms loaded,
- using `helm` charts and `kind` for a pseudo federated kubernetes environment,
- having only the controller endpoints exposed and
- the tests contained can only be of production nature.

### SMPC Environment

These tests are run:

- with the basic (monetdb, rabbitmq) docker images pre-built,
- with 3 workers(globalworker, 2 localworkers) and 1 controller,
- with the test data loaded through the mip_db container,
- with the production algorithms loaded,
- using `helm` charts and `kind` for a pseudo federated kubernetes environment,
- having only the controller endpoints exposed and
- the tests contained can only be of production nature.

### Algorithm Validation Tests

These tests are run:

- with the basic (monetdb, rabbitmq) docker images pre-built,
- with 1 localworker, 1 globalworker and 1 controller,
- with 10 localworkers, 1 globalworker and 1 controller in a different job,
- with the test data loaded using `inv load-data` and
- with the production algorithms loaded,
- using the `inv deploy` command of the `tasks.py` with a `.deployment.toml` template,
- can be run based on the workers' information in the `.deployment.toml`.
