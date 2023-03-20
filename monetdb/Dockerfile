FROM madgik/mipenginedb_base:0.1

#######################################################
# Download monetdb source files
#######################################################
RUN wget --output-document=/home/MonetDB-11.45.13.tar.bz2 --no-check-certificate https://www.monetdb.org/downloads/sources/Sep2022-SP2/MonetDB-11.45.13.tar.bz2
RUN tar -xf /home/MonetDB-11.45.13.tar.bz2 -C /home/

#######################################################
# Install monetdb
#######################################################
RUN pip3 install numpy==1.24.1 # Must be updated together with pyproject.toml
RUN mkdir /home/monetdb-build
WORKDIR /home/monetdb-build
RUN cmake -DCMAKE_BUILD_TYPE=Release -DASSERT=ON -DSTRICT=ON -DCMAKE_INSTALL_PREFIX=/usr/local/bin/monetdb /home/MonetDB-11.45.13
RUN cmake --build .
RUN cmake --build . --target install
ENV PATH="/usr/local/bin/monetdb/bin:$PATH"

EXPOSE 50000

#######################################################
# Installation clean up
#######################################################
RUN rm /home/MonetDB-11.45.13.tar.bz2
RUN rm -rf /home/MonetDB-11.45.13/
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
COPY mipengine/udfgen/udfio.py /home/udflib/
ENV PYTHONPATH "/home/udflib/"
ENV LOG_LEVEL "INFO"

#######################################################
# Initializing the default users passwords, that will be configured with configure_users.sh
#######################################################
ENV ADMIN_PASSWORD='admin'
ENV EXECUTOR_PASSWORD='executor'
ENV GUEST_PASSWORD='guest'

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


WORKDIR /home
CMD ["sh", "-c", "/home/bootstrap.sh; tail -fn +1 $MONETDB_STORAGE/merovingian.log -fn +1 $MONIT_CONFIG_FOLDER/monit.log"]
