#!/bin/bash

#Download/uzip images and labels
d="../datasets/cifar-10" #uzip directory
mkdir -p $d && cd $d

f='cifar-10'
#echo 'downloading' $url$f '...'
kaggle competitions download -c $f
unzip -q $f -d $d && rm $f &
