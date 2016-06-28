#!/bin/bash

#Instructions: move this file to the Tutorials directory containing the
# the .ipynb files you want to convert to html and run it.
OIFS="$IFS"
IFS=$'\n'
for nbname in `ls *.ipynb`; do
  ipython nbconvert --to html --template full "$nbname"
  basenm=`basename "$nbname" .ipynb`
  newname=`echo $basenm | sed -e 's/ /_/g' | tr -d \(\),\-`
  mv "$basenm.html" "$newname.html"
  echo "Created $newname.html"
done
IFS="$OIFS"
