# MIP-Engine

## Installation
1. Install python <br/>
```
sudo apt install python3.8

sudo apt install python3-pip
```

### Controller API
1. Install requirements. <br/>
```
python3.8 -m pip install -r ./requirements/controller.txt 
```

2. Run the controller API. <br/>
```
export QUART_APP=mipengine/controller/api/app:app; python3.8 -m quart run
```


## Tests

1. Install requirements <br/>
```
sudo apt install python3.8
sudo apt install tox
```

2. Run the tests <br/>
```
tox
```