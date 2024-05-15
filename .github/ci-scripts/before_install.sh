#!/bin/bash

sudo apt-get update -qq -y && \
sudo apt-get install -qq -y \
gfortran libblas-dev liblapack-dev openmpi-bin openmpi-common openssh-client \
openssh-server libopenmpi3 libopenmpi-dev libsuitesparse-dev
cmake --version
gcc --version

#download chp source code
curl -o ./jupyter_notebooks/Tutorials/algorithms/advanced/chp.c https://www.scottaaronson.com/chp/chp.c
#compile chp
gcc -o ./jupyter_notebooks/Tutorials/algorithms/advanced/chp ./jupyter_notebooks/Tutorials/algorithms/advanced/chp.c