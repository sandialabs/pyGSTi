#!/bin/bash
#These need to be run prior to building wheels using pip

python -m pip install --upgrade pip
# see https://github.com/cvxgrp/cvxpy/issues/968 for numpy version
python -m pip install "numpy>=1.16.0"
# Cython must be pre-installed to build native extensions on pyGSTi install
python -m pip install cython
python -m pip install wheel flake8
