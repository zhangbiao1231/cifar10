#!/bin/bash

#Download/uzip images and labels
d="../datasets" #uzip directory
mkdir -p $d && cd $d
url=http://d2l-data.s3-accelerate.amazonaws.com/
f='cifar-10_tiny.zip'
echo 'downloading' $url$f '...'
curl -L $url$f -o $f -# && unzip -q $f -d $d && rm $f &

wait