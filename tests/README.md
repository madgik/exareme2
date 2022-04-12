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
- with 3 nodes(globalnode, 2 localnodes) and 1 controller,
- with the demo data loaded using `inv load-data` and
- with the production and testing algorithms loaded,
- using the `inv deploy` command of the `tasks.py` with a `.deployment.toml` template,
- can be run based on the nodes' information in the `.deployment.toml`.

### Prod Environment

These tests are run:

- with all (monetdb, rabbitmq, node, controller) docker images pre-built,
- with 4 nodes(globalnode, 3 localnodes) and 1 controller,
- with the demo data loaded using `inv load-data` combined with the backup logic,
- with the production algorithms loaded,
- using `helm` charts and `kind` for a pseudo federated kubernetes environment,
- having only the controller endpoints exposed and
- the tests contained can only be of prod_env_tests nature.

### Algorithm Validation Tests

These tests are run:

- with the basic (monetdb, rabbitmq) docker images pre-built,
- with 1 localnode, 1 globalnode and 1 controller,
- with 10 localnodes, 1 globalnode and 1 controller in a different job,
- with the demo data loaded using `inv load-data` and
- with the production algorithms loaded,
- using the `inv deploy` command of the `tasks.py` with a `.deployment.toml` template,
- can be run based on the nodes' information in the `.deployment.toml`.
