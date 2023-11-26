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
    POETRY_VERSION=1.2.2 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    MONETDB_ADMIN_USERNAME="admin" \
    MONETDB_LOCAL_USERNAME="executor" \
    MONETDB_LOCAL_PASSWORD="executor" \
    MONETDB_PUBLIC_USERNAME="guest" \
    MONETDB_PUBLIC_PASSWORD="guest" \
    CODE_PATH="/opt/code"
ENV PATH="$POETRY_HOME/bin:$CODE_PATH:$PATH"
WORKDIR $CODE_PATH

#######################################################
# Installing poetry and dependencies
#######################################################
RUN pip install poetry==$POETRY_VERSION
COPY poetry.lock pyproject.toml ./
RUN poetry install --no-dev

#######################################################
# Copying the codebase
#######################################################
COPY exareme2/ ./exareme2/
RUN rm -rf exareme2/controller

#######################################################
# Setup bootstrap file
#######################################################
COPY exareme2/node/bootstrap.sh /home/bootstrap.sh
RUN chmod 775 /home/bootstrap.sh

#######################################################
# Setup MonetDB credentials Volume
#######################################################
ENV CREDENTIALS_CONFIG_FOLDER=/opt/credentials
RUN mkdir $CREDENTIALS_CONFIG_FOLDER
VOLUME $CREDENTIALS_CONFIG_FOLDER

CMD ["sh", "-c", "/home/bootstrap.sh"]
