# MIP-Engine [![codecov](https://codecov.io/gh/madgik/MIP-Engine/branch/master/graph/badge.svg?token=SZQ9S269RP)](https://codecov.io/gh/madgik/MIP-Engine)

### Prerequisites

1. Install [python3.8](https://www.python.org/downloads/ "python3.8")

1. Install [poetry](https://python-poetry.org/ "poetry")
   It is important to install `poetry` in isolation, so follow the
   recommended installation method.

## Setup

#### Environment Setup

1. Install dependencies

   ```
   poetry install
   ```

1. Activate virtual environment

   ```
   poetry shell
   ```

1. *Optional* To install tab completion for `invoke` run  (replacing `bash` with your shell)

   ```
   source <(poetry run inv --print-completion-script bash)
   ```

1. _Optional_ `pre-commit` is included in development dependencies. To install hooks

   ```
   pre-commit install
   ```

#### Local Deployment

1. Create a deployment configuration file `.deployment.toml` using the following template

   ```
   ip = "172.17.0.1"
   log_level = "INFO"
   celery_log_level ="INFO"
   monetdb_image = "madgik/mipenginedb:dev1.2"

   [[nodes]]
   id = "globalnode"
   monetdb_port=50000
   rabbitmq_port=5670

   [[nodes]]
   id = "localnode1"
   monetdb_port=50001
   rabbitmq_port=5671

   [[nodes]]
   id = "localnode2"
   monetdb_port=50002
   rabbitmq_port=5672
   ```

   and then run the following command to create the node config files

   ```
   inv create-node-configs
   ```

1. Deploy everything with

   ```
   inv deploy --start-all
   ```

1. _Optional_ Load the data into the db with

   ```
   inv load-data
   ```

1. Attach to some service's stdout/stderr with

   ```
   inv attach --controller
   ```

   or

   ```
   inv attach --node <NODE-NAME>
   ```

1. Restart services with

   ```
   inv start-node --all && inv start-controller --detached
   ```

#### Local Deployment (without single configuration file)

1. Create the node configuration files inside the `./configs/nodes/` directory following the `./mipengine/node/config.toml` template.

1. Deploy everything with:

   ```
   inv deploy --start-all --monetdb-image madgik/mipenginedb:dev1.2 --celery-log-level info
   ```

#### Algorithm Run

1. Make a post request, _e.g._
   ```
   python test_post_request.py
   ```
