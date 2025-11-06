## Monetdb 11.53.13 (Mar2025-SP2) dockerized

### Build the base image

It is based on an `ubuntu:20.04` image and:

- Sets up timezone
- uses apt-update and apt-install to install all monetdb requirements

To build a new image you must be on the project root `Exareme2/`, then

```
docker build -t madgik/exareme2_db_base:<IMAGETAG> -f monetdb/DockerfileBaseImage .
```

### Build

In order to change the initial monetdb configuration, go to the `bootstrap.sh`.

To build a new image you must be on the project root `Exareme2/`, then

```
docker build -t <USERNAME>/exareme2_db:<IMAGETAG> -f monetdb/Dockerfile .
```

## Run

Then run with

```
docker run -d -p 50000:50000 --name <CONTAINERNAME> <USERNAME>/exareme2_db:<IMAGETAG>
```

Access container db with

```
docker exec -it <CONTAINERNAME> mclient db
```
