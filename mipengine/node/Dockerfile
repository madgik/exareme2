FROM python:3.8.11-slim-buster

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.1.7 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    CODE_PATH="/opt/code"

ENV PATH="$POETRY_HOME/bin:$CODE_PATH:$PATH"

WORKDIR $CODE_PATH

RUN pip install poetry==$POETRY_VERSION
COPY poetry.lock pyproject.toml ./
RUN poetry install --no-dev

COPY mipengine/ ./mipengine/

# TODO REMOVE
ENV NODE_IDENTIFIER="globalnode" \
    NODE_ROLE="GLOBALNODE" \
    LOG_LEVEL="INFO" \
    NODE_REGISTRY_IP="172.17.0.1" \
    NODE_REGISTRY_PORT="8500" \
    RABBITMQ_IP="172.17.0.1" \
    RABBITMQ_PORT="5670" \
    MONETDB_IP="172.17.0.1" \
    MONETDB_PORT="50000"

CMD ["celery", "-A", "mipengine.node.node", "worker", "-l", "INFO"]
