#!/bin/bash

brew update && \
brew install \
gfortran libblas-dev liblapack-dev openmpi-bin openmpi-common openssh-client \
openssh-server libopenmpi3 libopenmpi-dev libsuitesparse-dev
cmake --version
gcc --version