# Unit Test for Explanation Generator Module

## Build

To be able to run the unit test, first make sure that docker container is started.

1. create python evironment and activate it
```sh
$ conda create -n test_env python=3.8.12
$ conda activate test_env
```

2. install the packages
```sh
$ pip install -r requirements.txt
```

## Run

3. Run the unit test
```sh
$ pytest
```