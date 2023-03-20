#!/bin/bash
monetdbd stop /home/monetdb
rm /home/monetdb/.m*
rm -rf /home/monetdb/db
./bootstrap.sh
