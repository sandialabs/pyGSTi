#!/bin/bash
# This script needs to be run as admin

#Install suitesparse for UMFPACK, which cvxopt now needs...
echo "CHECK for UMFPACK DEBUG"
sudo find / -iname umfpack.h 2>/dev/null
apt-get install libsuitesparse-metis-dev
sudo find / -iname umfpack.h 2>/dev/null

echo "Checking if pdflatex install is needed"

if [ "$ReportA" == "True" ]; then
    apt-get -qq install texlive-full
fi

if [ "$Drivers" == "True" ]; then
    apt-get -qq install texlive-latex-base
fi

if [ "$ReportA" == "True" ] || [ "$Drivers" == "True" ]; then
    echo "Installing pdflatex requirements"
    pushd /usr/share/texmf-texlive/
    wget http://mirrors.ctan.org/install/macros/latex/contrib/etoolbox.tds.zip
    wget http://mirrors.ctan.org/install/macros/latex/contrib/adjustbox.tds.zip
    wget http://mirrors.ctan.org/install/macros/latex/contrib/collectbox.tds.zip
    wget http://mirrors.ctan.org/install/macros/latex/contrib/pdfcomment.tds.zip
    wget http://mirrors.ctan.org/install/macros/latex/contrib/datetime2.tds.zip
    wget http://mirrors.ctan.org/install/macros/generic/tracklang.tds.zip
    wget http://mirrors.ctan.org/install/macros/latex/contrib/bezos.tds.zip
    wget http://mirrors.ctan.org/install/macros/latex/contrib/hyperref.tds.zip
    wget http://mirrors.ctan.org/install/macros/latex/contrib/oberdiek.tds.zip
    wget http://mirrors.ctan.org/install/macros/generic/ifxetex.tds.zip
    wget http://mirrors.ctan.org/install/macros/latex/contrib/standalone.tds.zip
    unzip -o etoolbox.tds.zip 
    unzip -o adjustbox.tds.zip 
    unzip -o collectbox.tds.zip 
    unzip -o pdfcomment.tds.zip 
    unzip -o datetime2.tds.zip 
    unzip -o tracklang.tds.zip 
    unzip -o bezos.tds.zip 
    unzip -o hyperref.tds.zip 
    unzip -o oberdiek.tds.zip 
    unzip -o ifxetex.tds.zip 
    unzip -o standalone.tds.zip 
    texhash  
    popd
else
    echo "pdflatex is not required for these tests (ReportA is not set to \"True\")"
fi

