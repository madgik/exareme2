FROM python:3.10.18-slim-bullseye
LABEL maintainer="Konstantinos Filippopolitis <kfilippopolitis@athenarc.gr>"

# Prevent pip/poetry issues in flaky networks
ENV PIP_NO_CACHE_DIR=yes
ENV PIP_DEFAULT_TIMEOUT=100

# Install poetry and configure it
RUN pip install --no-cache-dir poetry

WORKDIR /app

# Install only aggregation-server group dependencies
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false
RUN poetry install --with aggregation-server --no-root --no-interaction --no-ansi


COPY aggregation_server ./aggregation_server

EXPOSE 50051
ENV AGG_SERVER_CONFIG_FILE=/app/aggregation_server/config.toml

ENTRYPOINT ["python", "-m", "aggregation_server.server"]
