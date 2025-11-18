FROM python:3.10.18-slim-bullseye
MAINTAINER Thanasis Karampatsis <tkarabatsis@athenarc.gr>

#######################################################
# Setting up timezone
#######################################################
ENV TZ=Etc/GMT
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

#######################################################
# Setting up env variables and workdir
#######################################################
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.2.2 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    CODE_PATH="/opt/code"
ENV PATH="$POETRY_HOME/bin:$CODE_PATH:$PATH"
WORKDIR $CODE_PATH

#######################################################
# Installing poetry and dependencies
#######################################################
RUN pip install poetry==$POETRY_VERSION
COPY poetry.lock pyproject.toml ./
RUN poetry install --no-dev --without aggregation-server

#######################################################
# Copying the codebase
#######################################################
COPY exaflow/ ./exaflow/
RUN rm -rf exaflow/worker

#######################################################
# Define Hypercorn needed env variables and start hypercorn
#######################################################
ENV PYTHONPATH=$CODE_PATH
CMD ["sh", "-c", "hypercorn --config python:exaflow.controller.quart.hypercorn_config -b 0.0.0.0:5000 exaflow/controller/quart/app:app"]
