#!/bin/bash

brew update && \
brew install \
gfortran openblas lapack openmpi \
openssh suite-sparse
cmake --version
gcc --version

#download chp source code
curl -o ./jupyter_notebooks/Tutorials/algorithms/advanced/chp.c https://www.scottaaronson.com/chp/chp.c
#compile chp
gcc -o ./jupyter_notebooks/Tutorials/algorithms/advanced/chp ./jupyter_notebooks/Tutorials/algorithms/advanced/chp.c