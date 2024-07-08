#!/bin/bash

brew update && \
brew install \
gfortran openblas lapack openmpi \
openssh suite-sparse
cmake --version
gcc --version