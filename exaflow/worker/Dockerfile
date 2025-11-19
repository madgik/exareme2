FROM python:3.10.18-slim-bullseye
LABEL maintainer="Thanasis Karampatsis <tkarabatsis@athenarc.gr>"

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
    POETRY_VERSION=1.8.2 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    DATA_PATH="/opt/data" \
    CODE_PATH="/opt/code"
ENV PATH="$POETRY_HOME/bin:$CODE_PATH:$PATH"
WORKDIR $CODE_PATH

ENV PYTHONPATH=$CODE_PATH
RUN mkdir $DATA_PATH
VOLUME $DATA_PATH


#######################################################
# Installing poetry and dependencies
#######################################################
RUN pip install poetry==$POETRY_VERSION
COPY poetry.lock pyproject.toml ./
RUN poetry install --only main --no-root

#######################################################
# Copying the codebase
#######################################################
COPY exaflow/ ./exaflow/
RUN rm -rf exaflow/controller

#######################################################
# Setup bootstrap file
#######################################################
COPY exaflow/worker/bootstrap.sh /home/bootstrap.sh
RUN chmod 775 /home/bootstrap.sh

CMD ["sh", "-c", "/home/bootstrap.sh"]
