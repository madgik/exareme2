FROM python:3.8.11-slim-buster
MAINTAINER Thanasis Karampatsis <tkarabatsis@athenarc.gr>

#######################################################
# Setting up timezone
#######################################################
ENV TZ=Etc/GMT
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

#######################################################
# Setting up env variables
#######################################################
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    DATA_PATH="/opt/data" \
    DB_IP="172.17.0.1" \
    DB_PORT=50000 \
    DB_NAME="db" \
    SQLITE_DB_NAME="sqlite"

#######################################################
# Creating the data folder
#######################################################
RUN mkdir $DATA_PATH
VOLUME $DATA_PATH
WORKDIR $DATA_PATH

#######################################################
# Installing dependencies
#######################################################
RUN pip install mipdb==3.0.2 # Must be updated together with pyproject.toml
RUN pip install click==8.1.2
RUN pip install pymonetdb==1.6.3 # Must be updated together with pyproject.toml

#######################################################
# Initializing the default users passwords
#######################################################
ENV MONETDB_ADMIN_USERNAME="admin"
ENV MONETDB_LOCAL_USERNAME="executor"
ENV MONETDB_LOCAL_PASSWORD="executor"
ENV MONETDB_PUBLIC_USERNAME="guest"
ENV MONETDB_PUBLIC_PASSWORD="guest"

#######################################################
# Setup MonetDB credentials Volume
#######################################################
ENV CREDENTIALS_CONFIG_FOLDER=/opt/credentials
RUN mkdir $CREDENTIALS_CONFIG_FOLDER
VOLUME $CREDENTIALS_CONFIG_FOLDER

#######################################################
# Setup bootstrap files
#######################################################
COPY mipdb/bootstrap.sh /home/bootstrap.sh
RUN chmod 775 /home/bootstrap.sh

CMD ["sh", "-c", "/home/bootstrap.sh"]
