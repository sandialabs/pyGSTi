#!/bin/bash

brew update && \
brew install \
gfortran openblas lapack openmpi \
openssh suite-sparse
cmake --version
gcc --version

# Get the SuiteSparse source to allow compiling cvxopt when wheel is not available
# Not sure why brew install is not working for macos-11/Python 3.11, but it isn't
git clone https://github.com/DrTimothyAldenDavis/SuiteSparse.git
pushd SuiteSparse
git checkout v7.5.1
popd
export CVXOPT_SUITESPARSE_SRC_DIR=$(pwd)/SuiteSparse