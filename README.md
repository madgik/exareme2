# MIP-Engine

### Prerequisites

1. Install [python3.8](https://www.python.org/downloads/ "python3.8")

1. Install [poetry](https://python-poetry.org/ "poetry")
   It is important to install `poetry` in isolation, so follow the
   recomended installation method.

## Setup

#### Environment Setup

1. Install dependecies

   ```
   poetry install
   ```

1. Activate virtual environment

   ```
   poetry shell
   ```

#### Local Deployment

1. Find your machine's local ip address, *e.g.* with

   ```
   ifconfig | grep "inet "
   ```

1. Deploy everything

   ```
   invoke deploy --ip <YOUR-IP> --start-services
   ```

   *CAVEATS* The `--start-services` flag will start Controller (`quart`) and Nodes (`celery`). These
   processes will then run in the background. You can then manually kill them or use

   ```
   invoke killall-quart
   invoke killall-celery
   ```

   To see all available `invoke` tasks

   ```
   invoke --list
   ```

### Algorithm Run

1. Make a post request, *e.g.*
   ```
   python test_post_request.py
   ```
