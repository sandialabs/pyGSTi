#!/bin/bash

sudo apt-get update -qq -y && \
sudo apt-get install -qq -y \
gfortran libblas-dev liblapack-dev openmpi-bin openmpi-common openssh-client \
openssh-server libopenmpi1.10 libopenmpi-dev libsuitesparse-dev
cmake --version
gcc --version