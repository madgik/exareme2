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
    DB_PORT=50000

#######################################################
# Creating the data folder
#######################################################
RUN mkdir $DATA_PATH
VOLUME $DATA_PATH
WORKDIR $DATA_PATH

#######################################################
# Installing dependencies
#######################################################
RUN pip install mipdb==0.4.0
RUN pip install click==8.1.2
RUN pip install pymonetdb==1.4.1

#######################################################
# COPY log aggregator inside
#######################################################
COPY federation_info.py /home/federation_info.py

#######################################################
# Keep the container running
#######################################################
CMD tail -f /dev/null
