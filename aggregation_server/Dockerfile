# aggregation_server/Dockerfile
FROM python:3.8-slim

# 1) install Poetry
RUN pip install --no-cache-dir poetry

WORKDIR /app

# 2) bring in the PROJECT-LEVEL poetry files
COPY pyproject.toml poetry.lock ./

# 3) install *only* the aggregation-server deps
#    (we use the group you defined in root pyproject.toml)
RUN poetry config virtualenvs.create false \
 && poetry install --no-root --no-interaction --no-ansi --with aggregation-server

# 4) copy just the aggregation_server package code
COPY aggregation_server ./aggregation_server

EXPOSE 50051
ENV AGG_SERVER_CONFIG_FILE=/app/aggregation_server/config.toml

# since we installed into the container env (no venv), just run with python
ENTRYPOINT ["python", "-m", "aggregation_server.server"]
