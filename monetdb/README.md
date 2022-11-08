## Monetdb 11.45.7 (Sep2022) dockerized

### Build the base image

It is based on an `ubuntu:20.04` image and:

- Sets up timezone
- uses apt-update and apt-install to install all monetdb requirements

To build a new image you must be on the project root `MIP-Engine/`, then

```
docker build -t madgik/mipenginedb_base:<IMAGETAG> -f monetdb/DockerfileBaseImage .
```

### Build

In order to change the initial monetdb configuration, go to the `bootstrap.sh`.

To build a new image you must be on the project root `MIP-Engine/`, then

```
docker build -t <USERNAME>/mipenginedb:<IMAGETAG> -f monetdb/Dockerfile .
```

## Run

Then run with

```
docker run -d -p 50000:50000 --name <CONTAINERNAME> <USERNAME>/mipenginedb:<IMAGETAG>
```

Access container db with

```
docker exec -it <CONTAINERNAME> mclient db
```
