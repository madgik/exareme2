## Monetdb 11.39.13 (Oct2020-SP3) dockerized

### Build

In order to change the initial monetdb configuration, go to the `bootstrap.sh`.

To build a new image you must be on the project root `MIP-Engine/`, then

```
docker build -t <USERNAME>/mipenginedb:<IMAGETAG> -f monetdb/Dockerfile .
```

## Run

Then run with

```
docker run -d -P -p 50000:50000 --name <CONTAINERNAME> <USERNAME>/mipenginedb:<IMAGETAG>
```

Access container db with

```
docker exec -it <CONTAINERNAME> mclient db
```
