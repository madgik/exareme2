FROM madgik/exareme2_db_base:0.3

#######################################################
# Download monetdb source files
#######################################################
ARG MONETDB_VERSION=11.53.13
ARG MONETDB_RELEASE=Mar2025-SP2
ENV MONETDB_VERSION=${MONETDB_VERSION}
ENV MONETDB_RELEASE=${MONETDB_RELEASE}
ENV MONETDB_DOWNLOAD_URL=https://www.monetdb.org/downloads/sources/${MONETDB_RELEASE}/MonetDB-${MONETDB_VERSION}.tar.bz2
RUN wget --output-document=/home/monetDB.tar.bz2 --no-check-certificate "$MONETDB_DOWNLOAD_URL"
RUN tar -xf /home/monetDB.tar.bz2 -C /home/
COPY monetdb/capi.c /home/MonetDB-11.53.13/sql/backends/monet5/UDF/capi/
COPY monetdb/cheader.h /home/MonetDB-11.53.13/sql/backends/monet5/UDF/capi/
COPY monetdb/cheader.text.h /home/MonetDB-11.53.13/sql/backends/monet5/UDF/capi/


#######################################################
# Install monetdb
#######################################################
COPY ./monetdb/YeSQL/ /home/YeSQL
COPY ./monetdb/pypy3.10-v7.3.13-linux64/ /home/pypy3.10-v7.3.13-linux64
RUN ls -lah /home
RUN export MONETDB_FOLDER="/home"
RUN export CURRENT=/home
RUN pip3 install cffi
WORKDIR /home/YeSQL/functions
RUN export LD_LIBRARY_PATH="$MONETDB_FOLDER/udfs/;$MONETDB_FOLDER/pypy3.10-v7.3.13-linux64/bin;$MONETDB_FOLDER/YeSQL/functions/"
RUN export PYTHONPATH="$MONETDB_FOLDER/YeSQL/functions;$MONETDB_FOLDER;$MONETDB_FOLDER/YeSQL/"
RUN /home/pypy3.10-v7.3.13-linux64/bin/pypy wrappers.py libwrappedudfs2.so
RUN python3.10 wrappers.py libwrappedudfs.so

RUN pip3 install numpy==1.24.1 # Must be updated together with pyproject.toml

WORKDIR ../../..
RUN mkdir /home/monetdb-build
WORKDIR /home/monetdb-build
RUN cmake -DCMAKE_BUILD_TYPE=Release -DASSERT=ON -DSTRICT=ON -DCMAKE_INSTALL_PREFIX=/usr/local/bin/monetdb /home/MonetDB-${MONETDB_VERSION}
RUN cmake --build .
RUN cmake --build . --target install
ENV PATH="/usr/local/bin/monetdb/bin:$PATH"

EXPOSE 50000

#######################################################
# Installation clean up
#######################################################
RUN rm /home/monetDB.tar.bz2
RUN rm -rf /home/MonetDB-${MONETDB_VERSION}
RUN rm -rf /home/monetdb-build

#######################################################
# Download and install libstreams library from bionic repo
#######################################################
RUN wget http://gr.archive.ubuntu.com/ubuntu/pool/universe/s/strigi/libstreams0v5_0.7.8-2.2_amd64.deb http://gr.archive.ubuntu.com/ubuntu/pool/universe/s/strigi/libstreams-dev_0.7.8-2.2_amd64.deb
RUN dpkg -i libstreams0v5_0.7.8-2.2_amd64.deb libstreams-dev_0.7.8-2.2_amd64.deb

#######################################################
# Setup bootstrap files
#######################################################
COPY monetdb/bootstrap.sh /home/bootstrap.sh
COPY monetdb/configure_users.sh /home/configure_users.sh
COPY monetdb/configure_monit.sh /home/configure_monit.sh
COPY monetdb/reset_database.sh /home/reset_database.sh
RUN chmod 775 /home/bootstrap.sh
RUN chmod 775 /home/configure_users.sh
RUN chmod 775 /home/configure_monit.sh
RUN chmod 775 /home/reset_database.sh

#######################################################
# Setup logrotate file
#######################################################
COPY monetdb/logrotate.conf /etc/logrotate.d/monetdb
RUN chmod 444 /etc/logrotate.d/monetdb

#######################################################
# Install python libraries
#######################################################
RUN pip3 install scipy==1.10.0 # Must be updated together with pyproject.toml
RUN pip3 install pandas==1.5.2 # Must be updated together with pyproject.toml
RUN pip3 install scikit-learn==1.2.0 # Must be updated together with pyproject.toml
RUN pip3 install statsmodels==0.13.2 # Must be updated together with pyproject.toml

#######################################################
# Add /home/udflib to python path and copy
# necessary tools
#######################################################
COPY exareme2/algorithms/exareme2/udfgen/udfio.py /home/udflib/
ENV PYTHONPATH="/home/udflib/"
ENV LOG_LEVEL="INFO"

#######################################################
# Initializing the default users passwords, that will be configured with configure_users.sh
#######################################################
ENV MONETDB_ADMIN_USERNAME="admin"
ENV MONETDB_LOCAL_USERNAME="executor"
ENV MONETDB_LOCAL_PASSWORD="executor"
ENV MONETDB_PUBLIC_USERNAME="guest"
ENV MONETDB_PUBLIC_PASSWORD="guest"

#######################################################
# Setup MONETDB Volume
#######################################################
ENV MONETDB_NCLIENTS=64
ENV MONETDB_STORAGE=/home/monetdb
VOLUME $MONETDB_STORAGE

#######################################################
# Configure monit
#######################################################
ENV MONIT_CONFIG_FOLDER=/home/monit
RUN mkdir $MONIT_CONFIG_FOLDER

#######################################################
# Setup credentials Volume
#######################################################
ENV CREDENTIALS_CONFIG_FOLDER=/opt/credentials
RUN mkdir $CREDENTIALS_CONFIG_FOLDER
VOLUME $CREDENTIALS_CONFIG_FOLDER

# MAX monetdb memory in bytes
ENV MAX_MEMORY=2147483648
# Soft monetdb restart memory limit in megabytes
ENV SOFT_RESTART_MEMORY_LIMIT=1200
# Hard monetdb restart memory limit in megabytes
ENV HARD_RESTART_MEMORY_LIMIT=1600

WORKDIR /home
CMD ["sh", "-c", "/home/bootstrap.sh; tail -fn +1 $MONETDB_STORAGE/merovingian.log -fn +1 $MONIT_CONFIG_FOLDER/monit.log"]
