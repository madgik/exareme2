## Monetdb 11.37.11 (June-SP1) dockerized

### Build
In order to change the initial monetdb configuration, go to the `bootstrap.sh` inside the *config* folder.

Then you can build the image with:
```
docker build username/imagename:imagetag .
```

## Run

If you haven't built your own image download the latest from:
```
thanasulas/monetdb:11.37.11
```

Then you can run it with:
```
docker run -d -P -p 50000:50000 --name monetdb-1 thanasulas/monetdb:11.37.11
```

If you want to access monetdb inside the container:
```
docker exec -it monetdb-1 mclient db
```