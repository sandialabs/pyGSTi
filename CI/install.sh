#!/bin/bash
# This script needs to be run as admin
set -o verbose
echo "Beginning install.sh"
sudo apt-get update  > /dev/null 2>&1
echo "Update complete"

##An example of how to search for a file in apt packages
## (useful for debugging TravisCI build errors)
#apt-get install apt-file
#apt-file update
#apt-file search myFile.txt
#find / -iname myFile.txt 2>/dev/null

#Install suitesparse for UMFPACK development (e.g. umfpack.h)
# which is now needed by cvxopt. The second "cp" line is a
# COMPLETE HACK which is due to the cvxopt build thinking that
# it needs to link to a "libsuitesparseconfig.so" library that
# doesn't seem to exist.  So the below line simply duplicates
# lapack as this requested library, which allow the build to
# proceed since it apparently doesn't actually require anything
# in the non-existent library...
sudo apt-get install libsuitesparse-dev  > /dev/null 2>&1
sudo cp /usr/lib/liblapack.so /usr/lib/libsuitesparseconfig.so  > /dev/null 2>&1
echo "SuiteSparse complete"

cmake --version
gcc --version

# ----------------------------------------------------------------------
# The below block was used in TravisCI's 'precise' environment, when
# newer versions of gcc, g++, and cmake were needed (so we built them)
# ----------------------------------------------------------------------
#sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test > /dev/null 2>&1
#sudo apt-get update > /dev/null 2>&1
#
#echo "Reinstalling gcc/g++ to get newer versions"
#sudo update-alternatives --remove-all gcc  > /dev/null 2>&1
#sudo update-alternatives --remove-all g++  > /dev/null 2>&1
#sudo apt-get install gcc-4.8  > /dev/null 2>&1
#sudo apt-get install g++-4.8  > /dev/null 2>&1
#sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 20  > /dev/null 2>&1
#sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 20  > /dev/null 2>&1
#sudo update-alternatives --config gcc  > /dev/null 2>&1
#sudo update-alternatives --config g++  > /dev/null 2>&1
#sudo apt-get update  > /dev/null 2>&1
#sudo apt-get upgrade -y  > /dev/null 2>&1
#sudo apt-get dist-upgrade  > /dev/null 2>&1
#
#export CXX=g++
#echo "gcc/g++ install complete"
#
#cmake --version
#sudo apt remove cmake  > /dev/null 2>&1
#
## Install the following version of CMAKE
#echo "Installing newer version of cmake"
#version=3.11  > /dev/null 2>&1
#build=1  > /dev/null 2>&1
#mkdir ~/temp  > /dev/null 2>&1
#cd ~/temp  > /dev/null 2>&1
#wget https://cmake.org/files/v$version/cmake-$version.$build.tar.gz  > /dev/null 2>&1
#tar -xzvf cmake-$version.$build.tar.gz  > /dev/null 2>&1
#cd cmake-$version.$build/  > /dev/null 2>&1
#./bootstrap  > /dev/null 2>&1
#make -j4  > /dev/null 2>&1
#sudo make install  > /dev/null 2>&1
#cd ..  > /dev/null 2>&1
##rm -r temp  # > /dev/null 2>&1
#export PATH=/usr/local/bin:$PATH
#cmake --version
