FROM ubuntu:20.04
MAINTAINER Thanasis Karampatsis <tkarabatsis@athenarc.gr>

ENV LANG=C.UTF-8

#######################################################
# Setting up timezone
#######################################################
ENV TZ=Europe/Athens
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && apt install -y wget

#######################################################
# Download monetdb source files
#######################################################
RUN wget --output-document=/home/MonetDB-11.37.11.tar.bz2 --no-check-certificate https://www.monetdb.org/downloads/sources/Jun2020-SP1/MonetDB-11.37.11.tar.bz2
RUN tar -xf /home/MonetDB-11.37.11.tar.bz2 -C /home/
WORKDIR /home/MonetDB-11.37.11

#######################################################
# Install monetdb requirements
#######################################################

RUN apt update && apt install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt update && apt install -y python3.8-dev
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

RUN apt update && apt install -y libssl-dev libpcre3 libpcre3-dev pkg-config uuid-dev libxml2 libxml2-dev unixodbc-dev build-essential

RUN apt update && apt install -y python3-pip

RUN pip3 install numpy
#######################################################
# Install monetdb
#######################################################
RUN ./configure
RUN make
RUN make install

#######################################################
# Download and install libstreams library from bionic repo
#######################################################
RUN wget http://gr.archive.ubuntu.com/ubuntu/pool/universe/s/strigi/libstreams0v5_0.7.8-2.2_amd64.deb http://gr.archive.ubuntu.com/ubuntu/pool/universe/s/strigi/libstreams-dev_0.7.8-2.2_amd64.deb
RUN dpkg -i libstreams0v5_0.7.8-2.2_amd64.deb libstreams-dev_0.7.8-2.2_amd64.deb

#######################################################
# Setup bootstrap file
#######################################################
COPY monetdb_dockerized/config/bootstrap.sh /home/bootstrap.sh
RUN chmod 775 /home/bootstrap.sh

EXPOSE 50000

#######################################################
# Install python libraries
#######################################################
RUN pip3 install scipy
RUN pip3 install pandas

#######################################################
# Add /home/udflib to python path and copy
# necessary tools
#######################################################
COPY mipengine/node/udfgen/udfio.py /home/udflib/
ENV PYTHONPATH "/home/udflib/"

#######################################################
# Clean up
#######################################################
WORKDIR /home
RUN rm /home/MonetDB-11.37.11.tar.bz2 && rm -rf /home/MonetDB-11.37.11/

#######################################################
# DEV ONLY add username/password
#######################################################
RUN echo "user=monetdb" > /home/.monetdb
RUN echo "password=monetdb" >> /home/.monetdb

CMD ["/bin/bash","/home/bootstrap.sh"]
