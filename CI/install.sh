#!/bin/bash
# This script needs to be run as admin
sudo apt-get update

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
apt-get install libsuitesparse-dev
cp /usr/lib/liblapack.so /usr/lib/libsuitesparseconfig.so

sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update

sudo update-alternatives --remove-all gcc
sudo update-alternatives --remove-all g++
sudo apt-get install gcc-4.8
sudo apt-get install g++-4.8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 20
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 20
sudo update-alternatives --config gcc
sudo update-alternatives --config g++
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get dist-upgrade

export CXX=g++

sudo apt remove cmake

# Install the following version of CMAKE
version=3.11
build=1
mkdir ~/temp
cd ~/temp
wget https://cmake.org/files/v$version/cmake-$version.$build.tar.gz
tar -xzvf cmake-$version.$build.tar.gz
cd cmake-$version.$build/
./bootstrap
make -j4
sudo make install
cd ..
rm -r temp
cmake --version
