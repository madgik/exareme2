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
RUN poetry install --no-dev --without aggregation-server

#######################################################
# Copying the codebase
#######################################################
COPY exareme2/ ./exareme2/
RUN rm -rf exareme2/controller

#######################################################
# Setup bootstrap file
#######################################################
COPY exareme2/worker/bootstrap.sh /home/bootstrap.sh
RUN chmod 775 /home/bootstrap.sh

CMD ["sh", "-c", "/home/bootstrap.sh"]
