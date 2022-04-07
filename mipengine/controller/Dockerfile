FROM python:3.8.11-slim-buster
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
    POETRY_VERSION=1.1.7 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    CODE_PATH="/opt/code"
ENV PATH="$POETRY_HOME/bin:$CODE_PATH:$PATH"
WORKDIR $CODE_PATH


#######################################################
# Creating the cleanup file folder
#######################################################
ENV CLEANUP_FOLDER="/opt/cleanup"
RUN mkdir -p $CLEANUP_FOLDER
VOLUME $CLEANUP_FOLDER

#######################################################
# Installing poetry and dependencies
#######################################################
RUN pip install poetry==$POETRY_VERSION
COPY poetry.lock pyproject.toml ./
RUN poetry install --no-dev

#######################################################
# Copying the codebase
#######################################################
COPY mipengine/ ./mipengine/
RUN rm -rf mipengine/node

#######################################################
# Define Hypercorn neeeded env variables and start hypercorn
#######################################################
ENV FRAMEWORK_LOG_LEVEL="INFO"
ENV PYTHONPATH=$CODE_PATH
CMD ["sh", "-c", "hypercorn -b 0.0.0.0:5000 -w 1 --log-level $FRAMEWORK_LOG_LEVEL mipengine/controller/api/app:app"]
