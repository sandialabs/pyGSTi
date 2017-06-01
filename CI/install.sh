#!/bin/bash
# This script needs to be run as admin

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


#Latex is no longer needed!
#echo "Checking if pdflatex install is needed"
#
#if [ "$ReportA" == "True" ]; then
#    apt-get -qq install texlive-full
#fi
#
#if [ "$Drivers" == "True" ]; then
#    apt-get -qq install texlive-latex-base
#fi
#
#if [ "$ReportA" == "True" ] || [ "$Drivers" == "True" ]; then
#    echo "Installing pdflatex requirements"
#    pushd /usr/share/texmf-texlive/
#    wget http://mirrors.ctan.org/install/macros/latex/contrib/etoolbox.tds.zip
#    wget http://mirrors.ctan.org/install/macros/latex/contrib/adjustbox.tds.zip
#    wget http://mirrors.ctan.org/install/macros/latex/contrib/collectbox.tds.zip
#    wget http://mirrors.ctan.org/install/macros/latex/contrib/pdfcomment.tds.zip
#    wget http://mirrors.ctan.org/install/macros/latex/contrib/datetime2.tds.zip
#    wget http://mirrors.ctan.org/install/macros/generic/tracklang.tds.zip
#    wget http://mirrors.ctan.org/install/macros/latex/contrib/bezos.tds.zip
#    wget http://mirrors.ctan.org/install/macros/latex/contrib/hyperref.tds.zip
#    wget http://mirrors.ctan.org/install/macros/latex/contrib/oberdiek.tds.zip
#    wget http://mirrors.ctan.org/install/macros/generic/ifxetex.tds.zip
#    wget http://mirrors.ctan.org/install/macros/latex/contrib/standalone.tds.zip
#    unzip -o etoolbox.tds.zip 
#    unzip -o adjustbox.tds.zip 
#    unzip -o collectbox.tds.zip 
#    unzip -o pdfcomment.tds.zip 
#    unzip -o datetime2.tds.zip 
#    unzip -o tracklang.tds.zip 
#    unzip -o bezos.tds.zip 
#    unzip -o hyperref.tds.zip 
#    unzip -o oberdiek.tds.zip 
#    unzip -o ifxetex.tds.zip 
#    unzip -o standalone.tds.zip 
#    texhash  
#    popd
#else
#    echo "pdflatex is not required for these tests (ReportA is not set to \"True\")"
#fi

