FROM python:3.8-slim
MAINTAINER Konstantinos Filippopolitis <kfilippopolitis@athenarc.gr>

RUN pip install --no-cache-dir poetry

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false \
 && poetry install --no-root --no-interaction --no-ansi --with aggregation-server

COPY aggregation_server ./aggregation_server

EXPOSE 50051
ENV AGG_SERVER_CONFIG_FILE=/app/aggregation_server/config.toml

ENTRYPOINT ["python", "-m", "aggregation_server.server"]
