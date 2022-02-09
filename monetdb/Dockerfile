FROM ubuntu:20.04
MAINTAINER Thanasis Karampatsis <tkarabatsis@athenarc.gr>
LABEL version="1.4"

ENV LANG=C.UTF-8

#######################################################
# Setting up timezone
#######################################################
ENV TZ=Etc/GMT
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && apt install -y wget

#######################################################
# Install monetdb requirements
#######################################################
RUN apt update && apt install -y software-properties-common
RUN apt update && apt install -y cmake
RUN apt update && apt install -y bison
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt update && apt install -y python3.8-dev
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

RUN apt update && apt install -y libssl-dev libpcre3 libpcre3-dev pkg-config uuid-dev libxml2 libxml2-dev unixodbc-dev build-essential logrotate

RUN apt update && apt install -y python3-pip
RUN pip3 install numpy

#######################################################
# Download monetdb source files
#######################################################
RUN wget --output-document=/home/MonetDB-11.41.11.tar.bz2 --no-check-certificate https://www.monetdb.org/downloads/sources/Jul2021-SP1/MonetDB-11.41.11.tar.bz2
RUN tar -xf /home/MonetDB-11.41.11.tar.bz2 -C /home/

#######################################################
# Install monetdb
#######################################################
RUN mkdir /home/monetdb-build
WORKDIR /home/monetdb-build
RUN cmake -DCMAKE_BUILD_TYPE=Release -DASSERT=ON -DSTRICT=ON -DCMAKE_INSTALL_PREFIX=/usr/local/bin/monetdb /home/MonetDB-11.41.11
RUN cmake --build .
RUN cmake --build . --target install
ENV PATH="/usr/local/bin/monetdb/bin:$PATH"

EXPOSE 50000

#######################################################
# Installation clean up
#######################################################
RUN rm /home/MonetDB-11.41.11.tar.bz2
RUN rm -rf /home/MonetDB-11.41.11/
RUN rm -rf /home/monetdb-build

#######################################################
# Download and install libstreams library from bionic repo
#######################################################
RUN wget http://gr.archive.ubuntu.com/ubuntu/pool/universe/s/strigi/libstreams0v5_0.7.8-2.2_amd64.deb http://gr.archive.ubuntu.com/ubuntu/pool/universe/s/strigi/libstreams-dev_0.7.8-2.2_amd64.deb
RUN dpkg -i libstreams0v5_0.7.8-2.2_amd64.deb libstreams-dev_0.7.8-2.2_amd64.deb

#######################################################
# Setup bootstrap file
#######################################################
COPY monetdb/bootstrap.sh /home/bootstrap.sh
RUN chmod 775 /home/bootstrap.sh

#######################################################
# Setup logrotate file
#######################################################
COPY monetdb/logrotate.conf /etc/logrotate.d/monetdb
RUN chmod 444 /etc/logrotate.d/monetdb

#######################################################
# Install python libraries
#######################################################
RUN pip3 install scipy
RUN pip3 install pandas
RUN pip3 install scikit-learn

#######################################################
# Add /home/udflib to python path and copy
# necessary tools
#######################################################
COPY mipengine/udfgen/udfio.py /home/udflib/
ENV PYTHONPATH "/home/udflib/"
ENV LOG_LEVEL "INFO"

#######################################################
# DEV ONLY add username/password
#######################################################
RUN echo "user=monetdb" > /home/.monetdb
RUN echo "password=monetdb" >> /home/.monetdb

#######################################################
# Setup MONETDB Volume
#######################################################
ENV MONETDB_STORAGE=/home/monetdb
VOLUME $MONETDB_STORAGE

WORKDIR /home
CMD ["/bin/bash","/home/bootstrap.sh"]
